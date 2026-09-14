#!/usr/bin/env node
// =============================================================================
// gen-models.mjs — generate src/lib/models.generated.json from the recipe SSOT
// -----------------------------------------------------------------------------
// SSOT: https://github.com/Atlas-Inf/sparkrun-recipes
//   (read-only mirror expected at /workspace/sparkrun-recipes/recipes on the host
//    that runs this script — that public repo is the single source of truth for
//    every supported model + its canonical `sparkrun run` command).
//
// Regenerate with:   node site/scripts/gen-models.mjs
//
// Output is a 3-level tree consumed by the model navigation UI:
//   [{ vendor, icon, subfamilies: [{ name, recipes: [{...}] }] }]
//   level 1: vendor  = top-level brand (Qwen/Gemma/Nemotron/Mistral/MiniMax/DeepSeek)
//   level 2: subfamily = the recipe directory (e.g. qwen3.8, qwen3.6, gemma4)
//   level 3: recipe  = one recipes/**/*.yaml file
//
// Every recipes/**/*.yaml MUST appear in the output. The generated tree's
// total recipe count is asserted to equal the number of recipe YAML files.
// No third-party deps: a tiny hand-rolled reader parses the (deliberately
// simple) recipe schema — top-level scalars, a `metadata:` block of scalars
// plus a `description: |` literal block, and a `defaults:` scalar block.
// =============================================================================

import { readdirSync, statSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, dirname, basename, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const RECIPES_ROOT =
  process.env.ATLAS_RECIPES_ROOT ||
  (existsSync(resolve(here, '../../../../sparkrun-recipes/recipes'))
    ? resolve(here, '../../../../sparkrun-recipes/recipes')
    : existsSync(resolve(here, '../../../sparkrun-recipes/recipes'))
    ? resolve(here, '../../../sparkrun-recipes/recipes')
    : '/workspace/sparkrun-recipes/recipes');
const SSOT_URL = 'https://github.com/Atlas-Inf/sparkrun-recipes';

const OUT = resolve(here, '..', 'src', 'lib', 'models.generated.json');

// --- recursive YAML file discovery ------------------------------------------
function walkYaml(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) out.push(...walkYaml(full));
    else if (entry.endsWith('.yaml') || entry.endsWith('.yml')) out.push(full);
  }
  return out;
}

// --- minimal recipe reader ---------------------------------------------------
// Returns: { top: {scalars...}, metadata: {scalars + description}, defaults: {} }
function parseRecipe(text) {
  const lines = text.split('\n');
  const top = {};
  const metadata = {};
  const defaults = {};

  let target = top;
  let inDescription = false;
  const descLines = [];

  for (const raw of lines) {
    if (inDescription) {
      if (raw.startsWith('    ') || raw.trim() === '') {
        descLines.push(raw.slice(4));
        continue;
      }
      inDescription = false;
      metadata.description = descLines.join('\n').trim();
    }

    const trimmed = raw.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    if (/^metadata:\s*$/.test(raw)) {
      target = metadata;
      continue;
    }
    if (/^defaults:\s*$/.test(raw)) {
      target = defaults;
      continue;
    }

    if (target === metadata && /^description:\s*\|\s*$/.test(trimmed)) {
      inDescription = true;
      continue;
    }

    const m = raw.match(/^\s*([a-zA-Z0-9_-]+):\s*(.*)$/);
    if (m) {
      const key = m[1];
      let val = m[2].trim();
      if (
        (val.startsWith('"') && val.endsWith('"')) ||
        (val.startsWith("'") && val.endsWith("'"))
      ) {
        val = val.slice(1, -1);
      }
      target[key] = val;
    }
  }

  if (inDescription && descLines.length) {
    metadata.description = descLines.join('\n').trim();
  }

  return { top, metadata, defaults };
}

// --- taxonomy helpers --------------------------------------------------------
const VENDOR_META = {
  Qwen: { order: 1, icon: 'qwen' },
  Gemma: { order: 2, icon: 'gemma' },
  Nemotron: { order: 3, icon: 'nemotron' },
  Mistral: { order: 4, icon: 'mistral' },
  MiniMax: { order: 5, icon: 'minimax' },
  DeepSeek: { order: 6, icon: 'deepseek' }
};

const FAMILY_DISPLAY = {
  'qwen3.8': 'Qwen3.8',
  'qwen3.6': 'Qwen3.6',
  'qwen3.5': 'Qwen3.5',
  'qwen3-next': 'Qwen3-Next',
  'qwen3-coder-next': 'Qwen3-Coder-Next',
  'qwen3-vl': 'Qwen3-VL',
  'gemma4': 'Gemma-4',
  'diffusion-gemma': 'Gemma Diffusion',
  'nemotron-3-nano': 'Nemotron-3 Nano',
  'nemotron-3-super': 'Nemotron-3 Super',
  'nemotron-3.5-lightning': 'Nemotron-3.5 Lightning',
  'mistral-small-4': 'Mistral-Small-4',
  'minimax-m2.7': 'MiniMax-M2.7',
  'deepseek-v4': 'DeepSeek-V4'
};

const VENDOR_OF_FAMILY = {
  'qwen3.8': 'Qwen',
  'qwen3.6': 'Qwen',
  'qwen3.5': 'Qwen',
  'qwen3-next': 'Qwen',
  'qwen3-coder-next': 'Qwen',
  'qwen3-vl': 'Qwen',
  'gemma4': 'Gemma',
  'diffusion-gemma': 'Gemma',
  'nemotron-3-nano': 'Nemotron',
  'nemotron-3-super': 'Nemotron',
  'nemotron-3.5-lightning': 'Nemotron',
  'mistral-small-4': 'Mistral',
  'minimax-m2.7': 'MiniMax',
  'deepseek-v4': 'DeepSeek'
};

function vendorOf(fam) {
  const v = VENDOR_OF_FAMILY[fam];
  if (!v) {
    throw new Error(
      `Unknown recipe subfamily '${fam}'. Register it in VENDOR_OF_FAMILY in site/scripts/gen-models.mjs.`
    );
  }
  return v;
}

function inferTopology(stem, top) {
  const maxNodes = Number(top.max_nodes || '1');
  if (maxNodes > 1 || stem.includes('-ep2') || stem.includes('-tp2')) {
    return 'ep2';
  }
  return 'single';
}

function cleanQuant(q) {
  if (!q) return 'FP8';
  const u = q.toUpperCase();
  if (u.includes('NVFP4')) return 'NVFP4';
  if (u.includes('FP8')) return 'FP8';
  if (u.includes('BF16')) return 'BF16';
  return q;
}

function shortDescription(desc) {
  if (!desc) return '';
  const first = desc.split('\n')[0].trim();
  const cleaned = first.replace(/^[^—–-]+[—–-]\s*/, '').trim();
  return cleaned || first;
}

function recipeDisplay(stem) {
  const parts = stem.split('-');
  const out = parts.map((p) => {
    const lp = p.toLowerCase();
    if (lp === 'nvfp4') return 'NVFP4';
    if (lp === 'fp8') return 'FP8';
    if (lp === 'bf16') return 'BF16';
    if (lp === 'ep2') return 'EP=2';
    if (lp === 'tp2') return 'TP=2';
    if (lp === 'mtp') return 'MTP';
    if (lp === 'vl') return 'VL';
    if (lp === 'it') return 'IT';
    if (lp === 'dspark') return 'DSpark';
    if (lp === 'flash') return 'Flash';
    if (lp === 'dense' || lp === 'single') return p[0].toUpperCase() + p.slice(1);
    // param-style tokens: 80b, a3b, a10b, a12b, 0.8b, 122b -> uppercase
    if (/^a?\d+(\.\d+)?b$/.test(lp)) return p.toUpperCase();
    // version-bearing family tokens stay as-is (qwen3.5, gemma, minimax...)
    return p[0].toUpperCase() + p.slice(1);
  });
  return out.join(' ');
}

function recipeRank(stem) {
  // Qwen3.8: 27B default ranks first, then latency/throughput, then Flash
  if (stem === 'qwen3.8-27b-nvfp4') return 0;
  if (stem === 'qwen3.8-27b-nvfp4-latency') return 1;
  if (stem === 'qwen3.8-27b-nvfp4-throughput') return 2;
  if (stem === 'qwen3.8-flash-next-nvfp4') return 3;
  if (stem === 'qwen3.8-flash-next-nvfp4-throughput') return 4;
  if (stem === 'qwen3.8-27b-nvfp4-unsloth') return 5;
  if (stem === 'qwen3.8-27b-nvfp4-unsloth-bfcl') return 6;

  // Nemotron: DSpark ranks first
  if (stem.includes('dspark')) return 0;
  return 10;
}

// --- main --------------------------------------------------------------------
const files = walkYaml(RECIPES_ROOT).sort();
if (files.length === 0) {
  console.error(`No recipe YAML files found under ${RECIPES_ROOT}`);
  process.exit(1);
}

// Build a 3-level tree: vendor -> subfamily (recipe dir) -> recipes.
const vendorMap = new Map(); // vendor -> { subfamilies: Map<famKey, {name,recipes[]}> }
let recipeCount = 0;

for (const file of files) {
  const text = readFileSync(file, 'utf8');
  const { top, metadata } = parseRecipe(text);
  const fam = basename(dirname(file)); // recipe directory == subfamily key
  const stem = basename(file).replace(/\.(ya?ml)$/, '');
  const topology = inferTopology(stem, top);
  const vendor = vendorOf(fam);

  // sparkrun requires --hosts (it errors "No hosts specified" otherwise).
  // Single-node recipes target one Spark -> localhost. EP=2/TP=2 recipes
  // span two nodes, so show a two-host placeholder the user fills in.
  const hostsArg =
    topology === 'single' ? '--hosts localhost' : '--hosts <spark-1>,<spark-2>';
  const recipe = {
    displayName: recipeDisplay(stem),
    hfId: top.model || '',
    params: metadata.model_params || '',
    quant: metadata.quantization || '',
    quantClean: cleanQuant(metadata.quantization || ''),
    topology,
    description: shortDescription(metadata.description),
    command: `sparkrun run @atlas/${stem} ${hostsArg}`,
    recipeStem: stem,
    recipeUrl: `${SSOT_URL}/blob/main/recipes/${fam}/${stem}.yaml`
  };

  if (!vendorMap.has(vendor)) {
    vendorMap.set(vendor, new Map());
  }
  const subMap = vendorMap.get(vendor);
  if (!subMap.has(fam)) {
    subMap.set(fam, {
      name: FAMILY_DISPLAY[fam] || fam,
      recipes: []
    });
  }
  subMap.get(fam).recipes.push(recipe);
  recipeCount++;
}

// Stable ordering: vendors by VENDOR_META.order, subfamilies by their dir key,
// recipes by prioritized rank then stem. This keeps the JSON (and the rendered nav) deterministic.
const vendors = [...vendorMap.entries()]
  .map(([vendor, subs]) => {
    const subfamilies = [...subs.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([, sf]) => {
        sf.recipes.sort((a, b) => {
          const rA = recipeRank(a.recipeStem);
          const rB = recipeRank(b.recipeStem);
          if (rA !== rB) return rA - rB;
          return a.recipeStem.localeCompare(b.recipeStem);
        });
        return sf;
      });
    return { vendor, icon: VENDOR_META[vendor].icon, subfamilies };
  })
  .sort((a, b) => VENDOR_META[a.vendor].order - VENDOR_META[b.vendor].order);

const json = JSON.stringify(vendors, null, 2) + '\n';
writeFileSync(OUT, json);

const emitted = vendors.reduce(
  (n, v) => n + v.subfamilies.reduce((m, s) => m + s.recipes.length, 0),
  0
);
if (emitted !== recipeCount || emitted !== files.length) {
  console.error(
    `Recipe count mismatch: yaml files=${files.length}, emitted=${emitted}. SSOT: ${SSOT_URL}`
  );
  process.exit(1);
}

const subCount = vendors.reduce((n, v) => n + v.subfamilies.length, 0);
console.log(
  `Wrote ${OUT}\n  ${files.length} recipes across ${subCount} subfamilies` +
    ` / ${vendors.length} vendors (SSOT: ${SSOT_URL})`
);
for (const v of vendors) {
  const n = v.subfamilies.reduce((m, s) => m + s.recipes.length, 0);
  console.log(`  - ${v.vendor} (${n}):`);
  for (const s of v.subfamilies) console.log(`      · ${s.name}: ${s.recipes.length}`);
}
