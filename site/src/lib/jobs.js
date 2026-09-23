// =============================================================================
// jobs.js — the job board: GitHub issues labeled `job`, filed through
// .github/ISSUE_TEMPLATE/job.yml.
// -----------------------------------------------------------------------------
// Shared by scripts/gen-jobs.mjs (build-time snapshot) and Jobs.svelte (live
// refresh in the browser), so the two can never parse an issue differently.
//
// SSOT for the form is job.yml. HEADINGS below must match its field `label`s
// exactly (GitHub renders each answer under `### <label>`), and FIELD_IDS must
// match its field `id`s (GitHub prefills a form field from `?<id>=` in the
// new-issue URL). Rename a field there, rename it here.
// =============================================================================

export const REPO = 'Atlas-Inf/atlas';
export const JOB_LABEL = 'job';
export const JOBS_API = `https://api.github.com/repos/${REPO}/issues?labels=${JOB_LABEL}&state=all&sort=updated&per_page=100`;
export const NEW_JOB_URL = `https://github.com/${REPO}/issues/new`;
export const ALL_JOBS_URL = `https://github.com/${REPO}/issues?q=is%3Aissue+label%3A${JOB_LABEL}`;

const HEADINGS = {
  kind: 'Kind',
  model: 'Model',
  what: "What's the job?",
  radius: 'Impact radius',
  radiusDetail: 'What else could it touch?',
  testers: "Who's waiting or willing to test?"
};

export const FIELD_IDS = {
  kind: 'kind',
  model: 'model',
  what: 'what',
  radius: 'radius',
  radiusDetail: 'radius-detail',
  testers: 'testers'
};

export const KINDS = ['New model support', 'Feature for a model we already support'];
export const RADII = ['Just this model', 'Shared code, other models could change', 'Not sure'];

// Finished jobs stay on the board this long, so a contributor sees their work land.
const DONE_WINDOW_DAYS = 30;

// Split an issue-form body into { heading: answer }. GitHub writes
// `_No response_` for an optional field left blank.
function sections(body) {
  const out = {};
  const parts = (body || '').replace(/\r\n/g, '\n').split(/^###\s+(.+)$/m);
  for (let i = 1; i < parts.length; i += 2) {
    const answer = parts[i + 1].trim();
    out[parts[i].trim()] = answer === '_No response_' ? '' : answer;
  }
  return out;
}

function radiusTone(radius) {
  if (radius.startsWith('Just')) return 'contained';
  if (radius.startsWith('Shared')) return 'shared';
  return 'unknown';
}

function statusOf(issue) {
  if (issue.state === 'closed') return 'done';
  return (issue.assignees || []).length > 0 ? 'taken' : 'open';
}

// One REST issue -> one card, or null when it does not belong on the board
// (a pull request carrying the label, or a job closed as not planned / stale).
export function toJob(issue, now = Date.now()) {
  if (!issue || issue.pull_request) return null;
  if (issue.state === 'closed') {
    if (issue.state_reason === 'not_planned') return null;
    const closed = Date.parse(issue.closed_at || '');
    if (!Number.isFinite(closed) || now - closed > DONE_WINDOW_DAYS * 86400e3) return null;
  }
  const s = sections(issue.body);
  const radius = s[HEADINGS.radius] || 'Not sure';
  return {
    number: issue.number,
    url: issue.html_url,
    title: (issue.title || '').replace(/^\s*\[job\]\s*/i, '').trim() || `Job #${issue.number}`,
    kind: s[HEADINGS.kind] || '',
    model: s[HEADINGS.model] || '',
    what: s[HEADINGS.what] || '',
    radius,
    radiusTone: radiusTone(radius),
    radiusDetail: s[HEADINGS.radiusDetail] || '',
    testers: s[HEADINGS.testers] || '',
    status: statusOf(issue),
    takenBy: (issue.assignees || []).map((a) => a.login),
    wants: (issue.reactions && issue.reactions['+1']) || 0,
    comments: issue.comments || 0,
    updated: issue.updated_at
  };
}

const STATUS_ORDER = { open: 0, taken: 1, done: 2 };

// Open first, then taken, then done; within a group, most-wanted first, then freshest.
export function toJobs(issues, now = Date.now()) {
  return (Array.isArray(issues) ? issues : [])
    .map((i) => toJob(i, now))
    .filter(Boolean)
    .sort(
      (a, b) =>
        STATUS_ORDER[a.status] - STATUS_ORDER[b.status] ||
        b.wants - a.wants ||
        String(b.updated).localeCompare(String(a.updated))
    );
}

// Issue title for a job posted from the site: "[job] <model>: <first line>".
// The post form sends it as `?title=` alongside one `?<field id>=` per answer.
export function jobTitle(model, what) {
  const first = (what || '').trim().split('\n')[0];
  return `[job] ${[(model || '').trim(), first].filter(Boolean).join(': ').slice(0, 90)}`;
}

// "@alice (GB10), @bob" -> [{login:'alice'}, {text:' (GB10), '}, {login:'bob'}],
// so the card links each person and keeps the hardware notes around them.
export function mentionParts(text) {
  const parts = [];
  let last = 0;
  for (const m of (text || '').matchAll(/@([A-Za-z0-9][A-Za-z0-9-]{0,38})/g)) {
    if (m.index > last) parts.push({ text: text.slice(last, m.index) });
    parts.push({ login: m[1] });
    last = m.index + m[0].length;
  }
  if (last < (text || '').length) parts.push({ text: text.slice(last) });
  return parts;
}
