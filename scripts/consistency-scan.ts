/**
 * Site-wide consistency scan.
 *
 * Uses the TypeScript AST for robust, source-accurate checks instead of fragile
 * string slicing. Checks are split into errors (block CI) and warnings
 * (require manual review but do not fail the scan).
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';
import ts from 'typescript';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, '..');

interface Finding {
  file: string;
  line: number;
  message: string;
}

const errors: Finding[] = [];
const warnings: Finding[] = [];

function reportError(file: string, line: number, message: string) {
  errors.push({ file: path.relative(root, file), line, message });
}

function reportWarning(file: string, line: number, message: string) {
  warnings.push({ file: path.relative(root, file), line, message });
}

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

function lineOf(source: string, pos: number): number {
  let line = 1;
  for (let i = 0; i < pos && i < source.length; i++) {
    if (source[i] === '\n') line++;
  }
  return line;
}

function parseSource(file: string): ts.SourceFile {
  return ts.createSourceFile(file, fs.readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true);
}

// ---------------------------------------------------------------------------
// 1. Core AC/PG/SA algorithm files must not call Math.random().
//    (gridworld.ts intentionally retains legacy unseeded helpers for older
//     chapters and is reviewed separately.)
// ---------------------------------------------------------------------------
const seededAlgorithmFiles = [
  path.join(root, 'src/lib/rl/actorCritic.ts'),
  path.join(root, 'src/lib/rl/policyGradient.ts'),
  path.join(root, 'src/lib/rl/stochasticApproximation.ts'),
];
for (const file of seededAlgorithmFiles) {
  if (!fs.existsSync(file)) continue;
  const source = parseSource(file);
  function visit(node: ts.Node) {
    if (
      ts.isPropertyAccessExpression(node) &&
      node.expression.kind === ts.SyntaxKind.Identifier &&
      (node.expression as ts.Identifier).text === 'Math' &&
      node.name.text === 'random'
    ) {
      reportError(file, lineOf(source.text, node.getStart(source)), 'Unseeded Math.random() call in core algorithm file.');
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
}

// ---------------------------------------------------------------------------
// 2. Discrete AC algorithms must reference bootstrapOnTruncation (AST-based).
// ---------------------------------------------------------------------------
const actorCriticFile = path.join(root, 'src/lib/rl/actorCritic.ts');
const acSource = parseSource(actorCriticFile);
const discreteNames = ['qac', 'a2c', 'offPolicyActorCritic', 'qBasedOffPolicyActorCritic'];
function visitFunctions(node: ts.Node) {
  if (ts.isFunctionDeclaration(node) && node.name && discreteNames.includes(node.name.text)) {
    const text = acSource.text.substring(node.pos, node.end);
    if (!text.includes('bootstrapOnTruncation')) {
      reportError(actorCriticFile, lineOf(acSource.text, node.getStart(acSource)), `${node.name.text} does not handle bootstrapOnTruncation`);
    }
  }
  ts.forEachChild(node, visitFunctions);
}
visitFunctions(acSource);

// ---------------------------------------------------------------------------
// 3. CI must run tests before build and use Node 24.
// ---------------------------------------------------------------------------
const deployYml = path.join(root, '.github/workflows/deploy.yml');
const deploySrc = fs.readFileSync(deployYml, 'utf8');
if (!deploySrc.includes('node-version: 24')) {
  reportError(deployYml, 1, 'CI should use node-version: 24');
}
if (!deploySrc.includes('npm run test:all')) {
  reportError(deployYml, 1, 'CI should run npm run test:all before build');
}
if (deploySrc.indexOf('npm run test:all') > deploySrc.indexOf('npm run build')) {
  reportError(deployYml, 1, 'Tests should run before build');
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
      reportError(actorCriticFile, 1, `${algo.name} on ${name} produced ${result.episodes.length} episodes, expected 5`);
    }
    if (result.policyAfterEpisode.length !== result.episodes.length) {
      reportError(actorCriticFile, 1, `${algo.name} on ${name}: policyAfterEpisode length ${result.policyAfterEpisode.length} != episodes ${result.episodes.length}`);
    }
  }
}

// ---------------------------------------------------------------------------
// 5. Warn about Math.random anywhere in src/lib/rl (legacy shared utilities).
// ---------------------------------------------------------------------------
const rlFiles = walk(path.join(root, 'src/lib/rl'), (p) => p.endsWith('.ts'));
for (const file of rlFiles) {
  const source = parseSource(file);
  function visit(node: ts.Node) {
    if (
      ts.isPropertyAccessExpression(node) &&
      node.expression.kind === ts.SyntaxKind.Identifier &&
      (node.expression as ts.Identifier).text === 'Math' &&
      node.name.text === 'random'
    ) {
      reportWarning(file, lineOf(source.text, node.getStart(source)), 'Unseeded Math.random() found in shared RL library.');
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
}

// ---------------------------------------------------------------------------
// 6. Warn about hard-coded horizon/maxSteps defaults of 30.
// ---------------------------------------------------------------------------
const pageFiles = walk(path.join(root, 'src/pages'), (p) => p.endsWith('.tsx') || p.endsWith('.ts'));
for (const file of new Set([...pageFiles, ...rlFiles])) {
  const text = fs.readFileSync(file, 'utf8');
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.includes('//')) continue;
    if (/\b(horizonH|horizon|maxSteps)\s*[:=]\s*30\b/.test(line)) {
      reportWarning(file, i + 1, 'Hard-coded horizon/maxSteps = 30 found');
    }
  }
}

// ---------------------------------------------------------------------------
// 7. Warn about questionable wording / labelling.
// ---------------------------------------------------------------------------
for (const file of pageFiles) {
  const text = fs.readFileSync(file, 'utf8');
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.includes('完整回合')) {
      reportWarning(file, i + 1, '“完整回合” wording found; verify it is not used in a continuing setting');
    }
    if (line.includes('当前策略价值') && line.includes('actionValueToStateValue')) {
      reportWarning(file, i + 1, 'max/greedy-derived value may be mislabeled as current policy value');
    }
  }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------
if (warnings.length > 0) {
  console.warn(`⚠️ Consistency scan produced ${warnings.length} warning(s) for manual review:`);
  for (const w of warnings) {
    console.warn(`  ${w.file}:${w.line} — ${w.message}`);
  }
}

if (errors.length > 0) {
  console.error(`\n❌ Consistency scan failed with ${errors.length} error(s):`);
  for (const issue of errors) {
    console.error(`  ${issue.file}:${issue.line} — ${issue.message}`);
  }
  process.exit(1);
}

console.log('✅ Consistency scan passed');
