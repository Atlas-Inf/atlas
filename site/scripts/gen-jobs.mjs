#!/usr/bin/env node
// =============================================================================
// gen-jobs.mjs — generate src/lib/jobs.generated.json from GitHub issues
// -----------------------------------------------------------------------------
// SSOT: issues on Atlas-Inf/atlas labeled `job` (filed via the Job issue form),
//   fetched with the `gh` CLI (GH_TOKEN in CI, the logged-in user locally).
//
// This is only the SNAPSHOT the page prerenders with. Jobs.svelte refreshes the
// list from the GitHub API in the visitor's browser, so a job filed after the
// last deploy still shows up. Parsing lives in src/lib/jobs.js, shared by both.
//
// BEST-EFFORT, like gen-stars: it MUST NEVER fail the build. On any error it
// keeps the existing generated file, or writes an empty board, and exits 0.
//
// Regenerate with:   node site/scripts/gen-jobs.mjs
// Output:            { jobs: [...], generated_date }
// =============================================================================

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { REPO, JOB_LABEL, toJobs } from '../src/lib/jobs.js';

const here = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(here, '..', 'src', 'lib', 'jobs.generated.json');

function write(obj) {
  writeFileSync(OUT, JSON.stringify(obj, null, 2) + '\n');
}

function keepOrEmpty(reason) {
  console.warn(`gen-jobs: ${reason}`);
  if (existsSync(OUT)) {
    try {
      JSON.parse(readFileSync(OUT, 'utf8'));
      console.warn('gen-jobs: keeping the existing jobs.generated.json');
      return;
    } catch {
      // unreadable: fall through and replace it
    }
  }
  write({ jobs: [], generated_date: null });
}

try {
  const raw = execFileSync(
    'gh',
    ['api', `repos/${REPO}/issues?labels=${JOB_LABEL}&state=all&sort=updated&per_page=100`],
    { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], maxBuffer: 16 * 1024 * 1024 }
  );
  const jobs = toJobs(JSON.parse(raw));
  write({ jobs, generated_date: new Date().toISOString().slice(0, 10) });
  console.log(`gen-jobs: ${jobs.length} job(s) on the board`);
} catch (err) {
  keepOrEmpty(err && err.stderr ? String(err.stderr).trim() : String(err && err.message ? err.message : err));
}
