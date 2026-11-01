// SPDX-License-Identifier: AGPL-3.0-only

//! Initial issued chunk budget, separate from the larger allocation envelope.

pub(super) fn initial_chunk_budget(
    configured: usize,
    arena: usize,
    idle: bool,
    glm_c4_sparse: bool,
) -> usize {
    // Decode-row allocation headroom is not part of the qualified sparse
    // prefill budget. Keep issued geometry independent of idle/busy admission.
    if glm_c4_sparse {
        return configured.min(arena);
    }
    if idle { arena } else { configured }
}

#[cfg(test)]
mod tests {
    use super::initial_chunk_budget;

    #[test]
    fn sparse_glm_idle_and_busy_respect_configured_budget() {
        for (configured, arena) in [(1024, 1028), (256, 260), (4, 8), (1, 5)] {
            for idle in [false, true] {
                assert_eq!(
                    initial_chunk_budget(configured, arena, idle, true),
                    configured,
                    "selected issued chunk must not borrow decode capacity: configured={configured}, arena={arena}, idle={idle}",
                );
            }
        }
        for idle in [false, true] {
            assert_eq!(initial_chunk_budget(1024, 512, idle, true), 512);
        }
    }

    #[test]
    fn ordinary_or_sparse_off_preserves_existing_solo_expansion() {
        for (configured, arena) in [(1024, 1028), (256, 260), (4, 8), (1, 5)] {
            assert_eq!(initial_chunk_budget(configured, arena, true, false), arena);
            assert_eq!(
                initial_chunk_budget(configured, arena, false, false),
                configured
            );
        }
    }
}
