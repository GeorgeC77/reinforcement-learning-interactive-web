/**
 * Site-wide consistency scan.
 *
 * Uses the TypeScript AST for robust, source-accurate checks. Checks are split
 * into errors (block CI) and warnings (require manual review but do not fail
 * the scan unless they exceed the baseline).
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';
import ts from 'typescript';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, '..');

const BASELINE_WARNING_COUNT = 2;

interface AllowlistEntry {
  file: string;
  function: string;
  reason: string;
}

interface Finding {
  file: string;
  line: number;
  message: string;
  function?: string;
  reason?: string;
}

interface Report {
  errors: Finding[];
  warnings: Finding[];
  allowlisted: Finding[];
  routeCount: number;
  testCoverage: {
    algorithmTests: string[];
    e2eProjects: string[];
    routeCount: number;
  };
}

const errors: Finding[] = [];
const warnings: Finding[] = [];
const allowlisted: Finding[] = [];

const allowedUnseededRandom: AllowlistEntry[] = [
  {
    file: 'src/lib/rl/gridworld.ts',
    function: 'sampleAction',
    reason: 'unseeded fallback of the shared sampling helper; UI demos pass a seeded rng',
  },
  {
    file: 'src/lib/rl/gridworld.ts',
    function: 'shuffleInPlace',
    reason: 'internal helper of asyncValueIteration; accepts an optional seeded rng and only falls back to Math.random without one',
  },
  {
    file: 'src/lib/rl/gridworld.ts',
    function: 'asyncValueIteration',
    reason: 'accepts an optional seeded rng; unseeded Math.random only when the caller does not provide one',
  },
  {
    file: 'src/lib/rl/gridworld.ts',
    function: 'runMCExploringStartsEpisodes',
    reason: 'accepts an optional seeded rng; unseeded Math.random only when the caller does not provide one',
  },
];

function reportError(file: string, line: number, message: string) {
  errors.push({ file: path.relative(root, file), line, message });
}

function reportWarning(file: string, line: number, message: string) {
  warnings.push({ file: path.relative(root, file), line, message });
}

function reportAllowlisted(file: string, line: number, message: string, fn: string, reason: string) {
  allowlisted.push({ file: path.relative(root, file), line, message, function: fn, reason });
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

function getEnclosingFunctionName(node: ts.Node): string | null {
  let curr: ts.Node | undefined = node;
  while (curr) {
    if (ts.isFunctionDeclaration(curr) && curr.name) {
      return curr.name.text;
    }
    if (ts.isMethodDeclaration(curr) && curr.name) {
      return curr.name.getText();
    }
    if (ts.isArrowFunction(curr)) {
      const parent = curr.parent;
      if (parent && ts.isVariableDeclaration(parent) && ts.isIdentifier(parent.name)) {
        return parent.name.text;
      }
    }
    curr = curr.parent;
  }
  return null;
}

// ---------------------------------------------------------------------------
// 1. Core AC/PG/SA algorithm files must not call Math.random().
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
// 2. Discrete AC algorithms must reference bootstrapOnTruncation.
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
// 5. src/lib/rl Math.random must be in allowlist.
// ---------------------------------------------------------------------------
const rlFiles = walk(path.join(root, 'src/lib/rl'), (p) => p.endsWith('.ts'));
for (const file of rlFiles) {
  const source = parseSource(file);
  const relFile = path.relative(root, file).replace(/\\/g, '/');
  function visit(node: ts.Node) {
    if (
      ts.isPropertyAccessExpression(node) &&
      node.expression.kind === ts.SyntaxKind.Identifier &&
      (node.expression as ts.Identifier).text === 'Math' &&
      node.name.text === 'random'
    ) {
      const line = lineOf(source.text, node.getStart(source));
      const fnName = getEnclosingFunctionName(node);
      const entry = allowedUnseededRandom.find(
        (e) => e.file === relFile && e.function === fnName
      );
      if (entry) {
        reportAllowlisted(
          file,
          line,
          `Unseeded Math.random() in ${fnName}`,
          fnName ?? '(unknown)',
          entry.reason
        );
      } else {
        reportError(
          file,
          line,
          `Unseeded Math.random() in ${fnName ?? '(unknown)'} is not in allowlist.`
        );
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
}

// ---------------------------------------------------------------------------
// 6. Hard-coded horizon/maxSteps defaults of 30 must be annotated.
// ---------------------------------------------------------------------------
const pageFiles = walk(path.join(root, 'src/pages'), (p) => p.endsWith('.tsx') || p.endsWith('.ts'));
const sourceFiles = new Set([...pageFiles, ...rlFiles]);
const horizonRegexDirect = /\b(horizonH|horizon|maxSteps)\s*(?::\s*number\s*)?(?:=|:)\s*30\b/;
const horizonRegexUseState = /\b(horizonH|horizon|maxSteps)\b[^/\n]*=\s*useState\s*\(\s*30\s*\)/;
for (const file of sourceFiles) {
  const text = fs.readFileSync(file, 'utf8');
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const prevLine = i > 0 ? lines[i - 1] : '';
    const isAllowed =
      line.includes('CONSISTENCY_ALLOW_DEFAULT_HORIZON') ||
      prevLine.includes('CONSISTENCY_ALLOW_DEFAULT_HORIZON');
    if (isAllowed) continue;
    if (horizonRegexDirect.test(line) || horizonRegexUseState.test(line)) {
      reportError(file, i + 1, 'Hard-coded horizon/maxSteps = 30 without CONSISTENCY_ALLOW_DEFAULT_HORIZON annotation');
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
// Build report.
// ---------------------------------------------------------------------------
const smokeSpec = path.join(root, 'e2e/smoke.spec.ts');
let routeCount = 0;
if (fs.existsSync(smokeSpec)) {
  const smokeText = fs.readFileSync(smokeSpec, 'utf8');
  const match = smokeText.match(/const routes\s*=\s*\[([\s\S]*?)\]/);
  if (match) {
    routeCount = (match[1].match(/['"]\//g) || []).length;
  }
}

const packageJson = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
const algorithmTests = Object.keys(packageJson.scripts || {}).filter((k) =>
  /^test:(sa|td|fa|pg|ac|smoke)$/.test(k)
);

let e2eProjects: string[] = [];
const playwrightConfig = path.join(root, 'playwright.config.ts');
if (fs.existsSync(playwrightConfig)) {
  const pwText = fs.readFileSync(playwrightConfig, 'utf8');
  const projectMatches = pwText.match(/name:\s*['"]([^'"]+)['"]/g);
  if (projectMatches) {
    e2eProjects = projectMatches.map((m) => m.replace(/name:\s*['"]/, '').replace(/['"]$/, ''));
  }
}

const report: Report = {
  errors,
  warnings,
  allowlisted,
  routeCount,
  testCoverage: {
    algorithmTests,
    e2eProjects,
    routeCount,
  },
};

fs.writeFileSync(path.join(root, 'consistency-report.json'), JSON.stringify(report, null, 2));

// ---------------------------------------------------------------------------
// Report to console.
// ---------------------------------------------------------------------------
if (allowlisted.length > 0) {
  console.log(`ℹ️ ${allowlisted.length} allowlisted finding(s):`);
  for (const a of allowlisted) {
    console.log(`  ${a.file}:${a.line} — ${a.message} (${a.reason})`);
  }
}

if (warnings.length > 0) {
  console.warn(`⚠️ Consistency scan produced ${warnings.length} warning(s) for manual review (baseline ${BASELINE_WARNING_COUNT}):`);
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

if (warnings.length > BASELINE_WARNING_COUNT) {
  console.error(`\n❌ Consistency scan failed: ${warnings.length} warnings exceed baseline ${BASELINE_WARNING_COUNT}.`);
  process.exit(1);
}

console.log('✅ Consistency scan passed');
