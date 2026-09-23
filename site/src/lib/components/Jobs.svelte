<script>
  // The job board. Prerenders from the build-time snapshot, then refreshes from
  // the GitHub API on load so a job filed a minute ago is already here. Posting
  // is a plain GET form onto GitHub's new-issue page: it prefills the Job form,
  // the poster presses Submit there, and it works with JavaScript off.
  import { onMount } from 'svelte';
  import snapshot from '$lib/jobs.generated.json';
  import {
    JOBS_API, NEW_JOB_URL, ALL_JOBS_URL, JOB_LABEL, FIELD_IDS, KINDS, RADII, toJobs, jobTitle, mentionParts
  } from '$lib/jobs.js';
  import { jobs as copy } from '$lib/data.js';

  let jobs = $state(snapshot.jobs || []);
  let live = $state(false);
  let kind = $state('all');
  let showDone = $state(false);
  let posting = $state(false);
  let draft = $state({ kind: KINDS[0], model: '', what: '', radius: RADII[2], radiusDetail: '', testers: '' });

  const byKind = $derived(
    jobs.filter((j) => kind === 'all' || (kind === 'model' ? j.kind === KINDS[0] : j.kind === KINDS[1]))
  );
  const shown = $derived(byKind.filter((j) => showDone || j.status !== 'done'));
  const doneCount = $derived(byKind.filter((j) => j.status === 'done').length);
  const openCount = $derived(jobs.filter((j) => j.status === 'open').length);
  const title = $derived(jobTitle(draft.model, draft.what));

  const statusText = (j) =>
    j.status === 'open' ? copy.status.open : j.status === 'done' ? copy.status.done : copy.status.taken;
  const cta = (j) => (j.status === 'open' ? copy.cta.open : j.status === 'done' ? copy.cta.done : copy.cta.taken);

  onMount(() => {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 8000);
    fetch(JOBS_API, { signal: ctl.signal, headers: { Accept: 'application/vnd.github+json' } })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((issues) => {
        jobs = toJobs(issues);
        live = true;
      })
      // Rate-limited or offline: the snapshot stays up, nothing to tell the visitor.
      .catch(() => {})
      .finally(() => clearTimeout(timer));
    return () => ctl.abort();
  });
</script>

<div id="jobs" class="jobs">
  <div class="jobs-head">
    <div>
      <h3 class="jobs-title">{copy.title} {#if openCount > 0}<span class="jobs-count mono">{openCount} open</span>{/if}</h3>
      <p class="jobs-sub">{copy.sub}</p>
    </div>
    <div class="jobs-actions">
      <button type="button" class="btn btn-primary" aria-expanded={posting} aria-controls="job-post" onclick={() => (posting = !posting)}>
        {posting ? copy.postClose : copy.postOpen}
      </button>
      <a class="btn btn-ghost" href={ALL_JOBS_URL} target="_blank" rel="noopener">{copy.allOnGithub}</a>
    </div>
  </div>

  <form id="job-post" class="job-post" class:is-open={posting} method="get" action={NEW_JOB_URL} target="_blank" rel="noopener">
    <input type="hidden" name="template" value="job.yml" />
    <input type="hidden" name="labels" value={JOB_LABEL} />
    <input type="hidden" name="title" value={title} />
    <label class="jp-field">
      <span>{copy.form.kind}</span>
      <select name={FIELD_IDS.kind} bind:value={draft.kind}>
        {#each KINDS as k}<option>{k}</option>{/each}
      </select>
    </label>
    <label class="jp-field">
      <span>{copy.form.model}</span>
      <input name={FIELD_IDS.model} bind:value={draft.model} placeholder="nvidia/Qwen3.8-27B-NVFP4" required />
    </label>
    <label class="jp-field jp-wide">
      <span>{copy.form.what}</span>
      <textarea name={FIELD_IDS.what} bind:value={draft.what} rows="2" placeholder={copy.form.whatHint} required></textarea>
    </label>
    <label class="jp-field">
      <span>{copy.form.radius}</span>
      <select name={FIELD_IDS.radius} bind:value={draft.radius}>
        {#each RADII as r}<option>{r}</option>{/each}
      </select>
    </label>
    <label class="jp-field">
      <span>{copy.form.testers}</span>
      <input name={FIELD_IDS.testers} bind:value={draft.testers} placeholder={copy.form.testersHint} />
    </label>
    {#if draft.radius !== RADII[0]}
      <label class="jp-field jp-wide">
        <span>{copy.form.radiusDetail}</span>
        <input name={FIELD_IDS.radiusDetail} bind:value={draft.radiusDetail} placeholder={copy.form.radiusDetailHint} />
      </label>
    {/if}
    <div class="jp-foot jp-wide">
      <button type="submit" class="btn btn-primary">{copy.form.submit}</button>
      <span class="jp-note">{copy.form.note}</span>
    </div>
  </form>

  <div class="jobs-filters" role="group" aria-label={copy.filterLabel}>
    {#each copy.filters as f}
      <button type="button" class="jobs-chip" aria-pressed={kind === f.id} onclick={() => (kind = f.id)}>{f.text}</button>
    {/each}
    {#if doneCount > 0}
      <button type="button" class="jobs-chip jobs-chip-quiet" aria-pressed={showDone} onclick={() => (showDone = !showDone)}>
        {showDone ? copy.hideDone : `${copy.showDone} (${doneCount})`}
      </button>
    {/if}
    <span class="jobs-fresh mono" title={live ? '' : `snapshot ${snapshot.generated_date || ''}`}>
      {live ? copy.fresh.live : copy.fresh.snapshot}
    </span>
  </div>

  {#if shown.length === 0}
    <div class="job-empty">
      <p>{copy.empty}</p>
      <button type="button" class="btn btn-secondary" onclick={() => (posting = true)}>{copy.postOpen}</button>
    </div>
  {:else}
    <div class="jobs-grid">
      {#each shown as j (j.number)}
        <article class="job-card" class:is-done={j.status === 'done'}>
          <div class="job-top">
            <span class="job-status job-status-{j.status}">
              {statusText(j)}{#if j.status === 'taken' && j.takenBy.length}{` · @${j.takenBy[0]}`}{/if}
            </span>
            {#if j.kind}<span class="job-kind">{j.kind === KINDS[0] ? copy.kindShort.model : copy.kindShort.feature}</span>{/if}
          </div>
          <h4><a href={j.url} target="_blank" rel="noopener">{j.title}</a></h4>
          {#if j.model}<div class="job-model mono">{j.model}</div>{/if}
          {#if j.what}<p class="job-what">{j.what}</p>{/if}
          <div class="job-radius job-radius-{j.radiusTone}" title={j.radiusDetail}>
            {j.radius}{#if j.radiusDetail && j.radiusTone !== 'contained'}<span class="job-radius-detail">{` · ${j.radiusDetail}`}</span>{/if}
          </div>
          {#if j.testers}
            <div class="job-testers">
              <span class="job-testers-label">{copy.testersLabel}</span>
              {#each mentionParts(j.testers) as p}{#if p.login}<a href="https://github.com/{p.login}" target="_blank" rel="noopener">@{p.login}</a>{:else}{p.text}{/if}{/each}
            </div>
          {/if}
          <div class="job-foot">
            <span class="job-meta mono">
              {#if j.wants > 0}{j.wants} {copy.wantThis}{/if}{#if j.wants > 0 && j.comments > 0}{' · '}{/if}{#if j.comments > 0}{j.comments} {copy.comments}{/if}
            </span>
            <a class="job-cta" href={j.url} target="_blank" rel="noopener">{cta(j)} →</a>
          </div>
        </article>
      {/each}
    </div>
  {/if}
</div>
