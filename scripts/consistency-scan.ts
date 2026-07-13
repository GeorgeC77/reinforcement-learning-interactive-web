/**
 * Site-wide consistency scan.
 *
 * Checks that are cheap and deterministic enough to run in CI.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, '..');

interface Issue {
  file: string;
  line: number;
  message: string;
}

const issues: Issue[] = [];

function walk(dir: string, filter: (p: string) => boolean): string[] {
  const out: string[] = [];
  for (const entry of fs.readdirSync(dir)) {
    const full = path.join(dir, entry);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) {
      out.push(...walk(full, filter));
    } else if (stat.isFile() && filter(full)) {
      out.push(full);
    }
  }
  return out;
}

function report(file: string, line: number, message: string) {
  issues.push({ file: path.relative(root, file), line, message });
}

// ---------------------------------------------------------------------------
// 1. No unseeded Math.random() in core RL algorithm files.
//    (Older shared utilities such as gridworld.ts intentionally retain
//     unseeded helpers for non-AC chapters and are out of scope here.)
// ---------------------------------------------------------------------------
const algorithmFiles = [
  path.join(root, 'src/lib/rl/actorCritic.ts'),
  path.join(root, 'src/lib/rl/policyGradient.ts'),
  path.join(root, 'src/lib/rl/stochasticApproximation.ts'),
];
for (const file of algorithmFiles) {
  if (!fs.existsSync(file)) continue;
  const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim().startsWith('//')) continue;
    if (line.includes('Math.random()')) {
      report(file, i + 1, 'Unseeded Math.random() found in algorithm code.');
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Discrete AC algorithms must reference bootstrapOnTruncation.
// ---------------------------------------------------------------------------
const actorCriticFile = path.join(root, 'src/lib/rl/actorCritic.ts');
const actorCriticSrc = fs.readFileSync(actorCriticFile, 'utf8');
const discreteNames = ['function qac', 'function a2c', 'function offPolicyActorCritic', 'function qBasedOffPolicyActorCritic'];
for (const name of discreteNames) {
  const idx = actorCriticSrc.indexOf(name);
  if (idx < 0) {
    report(actorCriticFile, 1, `Missing ${name}`);
    continue;
  }
  const block = actorCriticSrc.slice(idx, actorCriticSrc.indexOf('\n}', idx) + 2);
  if (!block.includes('bootstrapOnTruncation')) {
    report(actorCriticFile, actorCriticSrc.slice(0, idx).split('\n').length, `${name} does not handle bootstrapOnTruncation`);
  }
}

// ---------------------------------------------------------------------------
// 3. CI must run tests before build and use Node 24.
// ---------------------------------------------------------------------------
const deployYml = path.join(root, '.github/workflows/deploy.yml');
const deploySrc = fs.readFileSync(deployYml, 'utf8');
if (!deploySrc.includes('node-version: 24')) {
  report(deployYml, 1, 'CI should use node-version: 24');
}
if (!deploySrc.includes('npm run test:all')) {
  report(deployYml, 1, 'CI should run npm run test:all before build');
}
if (deploySrc.indexOf('npm run test:all') > deploySrc.indexOf('npm run build')) {
  report(deployYml, 1, 'Tests should run before build');
}

// ---------------------------------------------------------------------------
// 4. Algorithm smoke: policyAfterEpisode length equals episode count.
// ---------------------------------------------------------------------------
const { qac, a2c, offPolicyActorCritic, qBasedOffPolicyActorCritic } = await import(
  pathToFileURL(path.join(root, 'src/lib/rl/actorCritic.ts')).href
);
const { DEFAULT_CONFIG, EPISODIC_PATH_CONFIG } = await import(
  pathToFileURL(path.join(root, 'src/lib/rl/gridworld.ts')).href
);

const configs = [
  { name: 'DEFAULT_CONFIG', config: DEFAULT_CONFIG },
  { name: 'EPISODIC_PATH_CONFIG', config: EPISODIC_PATH_CONFIG },
];

for (const { name, config } of configs) {
  for (const algo of [qac, a2c, offPolicyActorCritic, qBasedOffPolicyActorCritic]) {
    const result = algo(config, {
      seed: 1,
      horizonH: 10,
      actorAlpha: 0.01,
      criticAlpha: 0.05,
      episodes: 5,
      epsilon: 0.5,
    });
    if (result.episodes.length !== 5) {
      report(actorCriticFile, 1, `${algo.name} on ${name} produced ${result.episodes.length} episodes, expected 5`);
    }
    if (result.policyAfterEpisode.length !== result.episodes.length) {
      report(actorCriticFile, 1, `${algo.name} on ${name}: policyAfterEpisode length ${result.policyAfterEpisode.length} != episodes ${result.episodes.length}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------
if (issues.length > 0) {
  console.error(`❌ Consistency scan failed with ${issues.length} issue(s):`);
  for (const issue of issues) {
    console.error(`  ${issue.file}:${issue.line} — ${issue.message}`);
  }
  process.exit(1);
}

console.log('✅ Consistency scan passed');
