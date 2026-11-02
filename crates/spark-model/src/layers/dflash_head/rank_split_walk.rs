// SPDX-License-Identifier: AGPL-3.0-only

//! Walking a rank's steps of a split propose (`rank_split`).

use super::*;

/// What a walk does at each step.
pub(crate) trait SplitOps {
    fn piece(&self, piece: Piece) -> Result<()>;
    fn swap(&self, swap: Swap) -> Result<()>;
}

/// How the runs of pieces between swaps execute.
pub(crate) enum Graphs<'a> {
    /// Launch every piece.
    Eager,
    /// Capture each run as one graph (pushed here) and launch it.
    Capture(&'a mut Vec<GraphHandle>),
    /// Launch the graph captured for each run (a zero handle runs eagerly).
    Replay(&'a [GraphHandle]),
}

/// Graphs a walk of `steps` captures: one per swap, plus one.
pub(crate) fn run_count(steps: &[Step]) -> usize {
    swaps_of(steps).len() + 1
}

/// Walk `steps` on `stream`: each run of pieces, then the swap after it.
/// Swaps are never captured (the pair exchange refuses a capturing stream).
pub(crate) fn walk(
    steps: &[Step],
    mut graphs: Graphs<'_>,
    gpu: &dyn GpuBackend,
    stream: u64,
    ops: &dyn SplitOps,
) -> Result<()> {
    let eager = |run: &[Step]| -> Result<()> {
        run.iter().try_for_each(|s| match s {
            Step::Run(piece) => ops.piece(*piece),
            Step::Swap(_) => unreachable!("a run holds pieces only"),
        })
    };
    // `split` drops the separators: swap `index - 1` precedes run `index`.
    let swaps = swaps_of(steps);
    for (index, run) in steps.split(|s| matches!(s, Step::Swap(_))).enumerate() {
        if index > 0 {
            ops.swap(swaps[index - 1])?;
        }
        match &mut graphs {
            Graphs::Eager => eager(run)?,
            Graphs::Replay(handles) => match handles.get(index) {
                Some(g) if g.0 != 0 => gpu.launch_graph(*g, stream)?,
                _ => eager(run)?,
            },
            Graphs::Capture(out) => {
                gpu.begin_capture(stream)?;
                if let Err(e) = eager(run) {
                    gpu.abort_capture_if_active(stream);
                    return Err(e);
                }
                let graph = gpu.end_capture(stream)?;
                out.push(graph);
                // An empty capture replays eagerly, as the unsplit path does.
                if graph.0 != 0 {
                    gpu.launch_graph(graph, stream)?;
                } else {
                    eager(run)?;
                }
            }
        }
    }
    Ok(())
}
