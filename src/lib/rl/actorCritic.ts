/**
 * Actor-Critic methods for Chapter 10.
 *
 * All discrete algorithms use a seeded RNG, a uniform horizon H, and record
 * every transition in ACUpdateRecord so the UI can play them step-by-step.
 */

import { mulberry32 } from './stochasticApproximation';
import {
  type GridWorldConfig,
  type Policy,
  type Action,
  type StateValues,
  step,
  sampleActionWithRng,
  isTerminal,
  stochasticTransition,
  solveLinearSystem,
} from './gridworld';

export type ACFrame = ACUpdateRecord;

export interface ACUpdateRecord {
  episode: number;
  time: number;

  state: number;
  action: Action;
  reward: number;
  nextState: number;
  nextAction?: Action;

  done: boolean;
  truncated: boolean;

  actorPolicyBefore: number[];
  actorPolicyAfter: number[];
  actorFullPolicyBefore?: Policy;
  actorFullPolicyAfter?: Policy;

  criticEstimateBefore: number;
  criticBootstrap: number;
  criticTarget: number;
  tdError: number;
  criticEstimateAfter: number;

  actorWeight: number;
  scoreGradient: number[];
  actorDelta: number[];

  qBefore?: number[][];
  qAfter?: number[][];

  vBefore?: number[];
  vAfter?: number[];

  targetPolicy?: number[];
  behaviorPolicy?: number[];
  targetProb?: number;
  behaviorProb?: number;
  rho?: number;
  rawRho?: number;
  usedRho?: number;
  wasClipped?: boolean;
  /** Whether the update bootstrapped on a truncated (time-limit) transition. */
  bootstrapUsed?: boolean;
  /** Expected bootstrap value for Q-based off-policy target. */
  expectedBootstrap?: number;
  /** QAC actor weight mode used for this update. */
  actorWeightMode?: 'raw-q' | 'advantage';
}

export interface ACEpisodeRecord {
  cumulativeReward: number;
  discountedReturn: number;
  episodeLength: number;
  success: boolean;
  truncated: boolean;
}

export interface ACResult {
  updates: ACUpdateRecord[];
  frames: ACFrame[];
  episodes: ACEpisodeRecord[];
  initialPolicy: Policy;
  policyAfterEpisode: Policy[];
  diverged: boolean;
  divergenceStep?: number;
  divergenceReason?: string;
  largeMagnitudeWarning?: { step: number; reason: string };
  finalQ?: number[][];
  finalV?: number[];
  finalPolicy?: Policy;
}

export interface ACOptions {
  seed: number;
  horizonH: number;
  actorAlpha: number;
  criticAlpha: number;
  episodes: number;
  epsilon?: number;
  /** If true (default), bootstrap at the artificial horizon truncation boundary. */
  bootstrapOnTruncation?: boolean;
  /** QAC only: use raw Q(s,a) or Q(s,a) - V_pi(s) as actor weight. */
  actorWeightMode?: 'raw-q' | 'advantage';
  /** Off-policy only: use raw importance ratios or clip them. */
  importanceMode?: 'raw' | 'clipped';
  /** Clipping threshold when importanceMode is 'clipped'. */
  clipThreshold?: number;
}

function softmaxDistribution(preferences: number[]): number[] {
  const maxPref = Math.max(...preferences);
  const exps = preferences.map((p) => Math.exp(p - maxPref));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function softmaxScore(policy: number[], action: number): number[] {
  return policy.map((p, a) => (a === action ? 1 : 0) - p);
}

function vectorL2(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

function finite(x: number): boolean {
  return Number.isFinite(x) && !Number.isNaN(x);
}

function allFinite(arr: number[]): boolean {
  return arr.every(finite);
}

const SAFE_THRESHOLDS = {
  tdError: 1e6,
  actorWeight: 1e6,
  criticEstimate: 1e9,
  actorGradientNorm: 1e6,
  criticUpdateNorm: 1e6,
};

export type NumericalHealth =
  | { status: 'stable' }
  | { status: 'warning'; reason: string }
  | { status: 'stopped'; reason: string };

export function checkNumericalHealth(record: ACUpdateRecord): NumericalHealth {
  if (!finite(record.tdError) || !finite(record.actorWeight)) {
    return { status: 'stopped', reason: 'non-finite TD error or actor weight' };
  }
  if (!allFinite(record.actorDelta) || !allFinite(record.scoreGradient)) {
    return { status: 'stopped', reason: 'non-finite actor gradient or score' };
  }
  if (!allFinite(record.actorPolicyBefore) || !allFinite(record.actorPolicyAfter)) {
    return { status: 'stopped', reason: 'non-finite policy probabilities' };
  }
  if (record.actorFullPolicyBefore && !record.actorFullPolicyBefore.every((dist) => allFinite(dist))) {
    return { status: 'stopped', reason: 'non-finite full policy probabilities' };
  }
  if (record.actorFullPolicyAfter && !record.actorFullPolicyAfter.every((dist) => allFinite(dist))) {
    return { status: 'stopped', reason: 'non-finite full policy probabilities' };
  }
  if (record.vBefore && !allFinite(record.vBefore)) {
    return { status: 'stopped', reason: 'non-finite value vector' };
  }
  if (record.vAfter && !allFinite(record.vAfter)) {
    return { status: 'stopped', reason: 'non-finite value vector' };
  }
  if (record.qBefore && !record.qBefore.every((row) => allFinite(row))) {
    return { status: 'stopped', reason: 'non-finite Q table' };
  }
  if (record.qAfter && !record.qAfter.every((row) => allFinite(row))) {
    return { status: 'stopped', reason: 'non-finite Q table' };
  }

  const actorGradientNorm = vectorL2(record.actorDelta);
  const criticUpdateNorm = Math.abs(record.criticEstimateAfter - record.criticEstimateBefore);
  const reasons: string[] = [];
  if (Math.abs(record.tdError) > SAFE_THRESHOLDS.tdError) reasons.push(`|TD error|=${record.tdError.toExponential(2)}`);
  if (Math.abs(record.actorWeight) > SAFE_THRESHOLDS.actorWeight) reasons.push(`|actor weight|=${record.actorWeight.toExponential(2)}`);
  if (
    Math.abs(record.criticEstimateBefore) > SAFE_THRESHOLDS.criticEstimate ||
    Math.abs(record.criticEstimateAfter) > SAFE_THRESHOLDS.criticEstimate
  ) {
    reasons.push(`|critic estimate| exceeds ${SAFE_THRESHOLDS.criticEstimate}`);
  }
  if (actorGradientNorm > SAFE_THRESHOLDS.actorGradientNorm) reasons.push(`actor gradient norm=${actorGradientNorm.toExponential(2)}`);
  if (criticUpdateNorm > SAFE_THRESHOLDS.criticUpdateNorm) reasons.push(`critic update norm=${criticUpdateNorm.toExponential(2)}`);

  if (reasons.length > 0) {
    return { status: 'warning', reason: reasons.join('; ') };
  }
  return { status: 'stable' };
}

export function checkCoverage(targetPolicy: number[], behaviorPolicy: number[]): void {
  for (let a = 0; a < targetPolicy.length; a++) {
    if (targetPolicy[a] > 0 && behaviorPolicy[a] <= 0) {
      throw new Error(
        `Coverage violation: target policy gives action ${a} probability ${targetPolicy[a]}, but behavior policy probability is ${behaviorPolicy[a]}.`
      );
    }
  }
}

export function policyWeightedStateValues(q: number[][], policy: Policy): StateValues {
  return q.map((qState, s) => policy[s].reduce((sum, p, a) => sum + p * qState[a], 0));
}

function computeEntropy(policy: number[]): number {
  return -policy.reduce((sum, p) => (p > 0 ? sum + p * Math.log(p) : sum), 0);
}

/**
 * KL(oldPolicy || newPolicy) = Σ_a oldPolicy[a] log(oldPolicy[a] / newPolicy[a]).
 *
 * Zero-probability handling:
 * - If oldPolicy[a] === 0, the term contributes 0.
 * - If oldPolicy[a] > 0 and newPolicy[a] === 0, the divergence is Infinity.
 */
export function klDivergence(oldPolicy: number[], newPolicy: number[]): number {
  let sum = 0;
  for (let a = 0; a < oldPolicy.length; a++) {
    const p = oldPolicy[a];
    const q = newPolicy[a];
    if (p === 0) continue;
    if (q <= 0) return Infinity;
    sum += p * Math.log(p / q);
  }
  return sum;
}

export function movingAverage(values: number[], window: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = values.slice(start, i + 1);
    out.push(slice.reduce((a, b) => a + b, 0) / slice.length);
  }
  return out;
}

export function effectiveSampleSize(rhos: number[]): number {
  const sum = rhos.reduce((a, b) => a + b, 0);
  const sumSq = rhos.reduce((a, b) => a + b * b, 0);
  return sumSq > 0 ? (sum * sum) / sumSq : 0;
}

export interface ACMetricSeries {
  episode: number;
  latestReturn: number;
  movingAverageReturn: number;
  overallAverageReturn: number;
  movingAverageEpisodeLength: number;
  overallAverageEpisodeLength: number;
  success: number;
  successRate: number;
  signedMeanTdError: number;
  meanAbsoluteTdError: number;
  rmsTdError: number;
  actorUpdateNorm: number;
  criticUpdateNorm: number;
  entropy: number;
  meanKL: number;
}

export function computeACMetricSeries(
  result: ACResult,
  window = 10
): ACMetricSeries[] {
  const N = result.episodes.length;
  if (N === 0) return [];

  const initialPolicy = result.initialPolicy;
  const policyAfterEpisode = result.policyAfterEpisode;

  const perEpisode: {
    tdErrors: number[];
    actorNorms: number[];
    criticNorms: number[];
    lengths: number[];
  }[] = Array.from({ length: N }, () => ({ tdErrors: [], actorNorms: [], criticNorms: [], lengths: [] }));

  result.updates.forEach((u) => {
    const e = u.episode;
    if (e >= 1 && e <= N) {
      perEpisode[e - 1].tdErrors.push(u.tdError);
      perEpisode[e - 1].actorNorms.push(vectorL2(u.actorDelta));
      perEpisode[e - 1].criticNorms.push(Math.abs(u.criticEstimateAfter - u.criticEstimateBefore));
    }
  });

  const returns = result.episodes.map((ep) => ep.cumulativeReward);
  const successes = result.episodes.map((ep) => (ep.success ? 1 : 0));
  const returnMA = movingAverage(returns, window);
  const successRateMA = movingAverage(successes, window);
  const lengthMA = movingAverage(result.episodes.map((ep) => ep.episodeLength), window);

  let cumulativeReturnSum = 0;
  let cumulativeLengthSum = 0;

  return result.episodes.map((ep, i) => {
    cumulativeReturnSum += ep.cumulativeReward;
    cumulativeLengthSum += ep.episodeLength;
    const beforePolicy = i === 0 ? initialPolicy : policyAfterEpisode[i - 1];
    const afterPolicy = policyAfterEpisode[i];
    const ent =
      afterPolicy.reduce((sum, dist) => sum + computeEntropy(dist), 0) / afterPolicy.length;
    const meanKL =
      afterPolicy.reduce((sum, dist, s) => sum + klDivergence(beforePolicy[s], dist), 0) /
      afterPolicy.length;

    const tdErrs = perEpisode[i].tdErrors;
    const signedMean = tdErrs.length ? tdErrs.reduce((a, b) => a + b, 0) / tdErrs.length : 0;
    const meanAbs = tdErrs.length ? tdErrs.reduce((a, b) => a + Math.abs(b), 0) / tdErrs.length : 0;
    const rms = tdErrs.length ? Math.sqrt(tdErrs.reduce((a, b) => a + b * b, 0) / tdErrs.length) : 0;

    return {
      episode: i + 1,
      latestReturn: ep.cumulativeReward,
      movingAverageReturn: returnMA[i],
      overallAverageReturn: cumulativeReturnSum / (i + 1),
      movingAverageEpisodeLength: lengthMA[i],
      overallAverageEpisodeLength: cumulativeLengthSum / (i + 1),
      success: ep.success ? 1 : 0,
      successRate: successRateMA[i],
      signedMeanTdError: signedMean,
      meanAbsoluteTdError: meanAbs,
      rmsTdError: rms,
      actorUpdateNorm: perEpisode[i].actorNorms.length
        ? perEpisode[i].actorNorms.reduce((a, b) => a + b, 0) / perEpisode[i].actorNorms.length
        : 0,
      criticUpdateNorm: perEpisode[i].criticNorms.length
        ? perEpisode[i].criticNorms.reduce((a, b) => a + b, 0) / perEpisode[i].criticNorms.length
        : 0,
      entropy: ent,
      meanKL,
    };
  });
}

export interface SampledTransition {
  nextState: number;
  reward: number;
  done: boolean;
}

export function sampleTransitionFromStateAction(
  state: number,
  action: Action,
  config: GridWorldConfig,
  _rng: () => number
): SampledTransition {
  const result = step(state, action, config);
  return { nextState: result.nextState, reward: result.reward, done: result.done };
}

export interface StateActionTdEstimate {
  samples: number[];
  mean: number;
  std: number;
  count: number;
}

/**
 * Sample a one-step TD error for a fixed (state, action) pair using the provided
 * value estimator and RNG.  In a deterministic GridWorld the standard deviation
 * is zero, but the interface is sample-based so the UI can show count/mean/std/error.
 */
export function sampleTdErrorAtStateAction(options: {
  config: GridWorldConfig;
  state: number;
  action: Action;
  values: number[];
  rng: () => number;
}): StateActionTdEstimate {
  const { config, state, action, values, rng } = options;
  if (isTerminal(state, config)) {
    return { samples: [], mean: 0, std: 0, count: 0 };
  }
  const transition = sampleTransitionFromStateAction(state, action, config, rng);
  const bootstrap = transition.done ? 0 : values[transition.nextState];
  const tdError = transition.reward + config.gamma * bootstrap - values[state];
  return { samples: [tdError], mean: tdError, std: 0, count: 1 };
}

/** Solve Bellman equations for state values under a stochastic transition model. */
function solveBellmanEquation(
  rewardVector: number[],
  transitionMatrix: number[][],
  gamma: number
): number[] {
  const n = rewardVector.length;
  // (I - gamma P) v = r
  const a: number[][] = Array.from({ length: n }, (_, i) =>
    transitionMatrix[i].map((p, j) => (i === j ? 1 - gamma * p : -gamma * p))
  );
  return solveLinearSystem(a, rewardVector);
}

/** Build policy-weighted rewards and transitions under the configured slip probability. */
function policyWeightedStochasticDynamics(
  policy: Policy,
  config: GridWorldConfig,
  slip: number
): { rewardVector: number[]; transitionMatrix: number[][] } {
  const n = config.rows * config.cols;
  const rewardVector = new Array(n).fill(0);
  const transitionMatrix = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let s = 0; s < n; s++) {
    if (isTerminal(s, config)) continue;
    for (let a = 0; a < policy[s].length; a++) {
      const piA = policy[s][a];
      if (piA === 0) continue;
      const dist = stochasticTransition(s, a as Action, config, slip);
      for (const { nextState, prob } of dist) {
        const { reward } = step(s, a as Action, config);
        rewardVector[s] += piA * prob * reward;
        transitionMatrix[s][nextState] += piA * prob;
      }
    }
  }
  return { rewardVector, transitionMatrix };
}

/** Compute true state values under a policy with stochastic transitions. */
export function solveStateValuesWithSlip(
  policy: Policy,
  config: GridWorldConfig,
  slip: number
): number[] {
  const { rewardVector, transitionMatrix } = policyWeightedStochasticDynamics(
    policy,
    config,
    slip
  );
  return solveBellmanEquation(rewardVector, transitionMatrix, config.gamma);
}

/** Compute true state-action Q-values under a policy with stochastic transitions. */
export function computeQValuesWithSlip(
  policy: Policy,
  config: GridWorldConfig,
  slip: number
): number[][] {
  const values = solveStateValuesWithSlip(policy, config, slip);
  const n = config.rows * config.cols;
  const q: number[][] = Array.from({ length: n }, () => new Array(policy[0]?.length ?? 5).fill(0));
  for (let s = 0; s < n; s++) {
    if (isTerminal(s, config)) continue;
    for (let a = 0; a < q[s].length; a++) {
      const dist = stochasticTransition(s, a as Action, config, slip);
      let exp = 0;
      for (const { nextState, prob } of dist) {
        const { reward, done } = step(s, a as Action, config);
        const bootstrap = done ? 0 : values[nextState];
        exp += prob * (reward + config.gamma * bootstrap);
      }
      q[s][a] = exp;
    }
  }
  return q;
}

/**
 * Build the matrix M[a][k] = π(a|s) * b(s,a) * score(s,a)[k].
 * Summing over actions gives the per-parameter contribution of the baseline.
 * For an action-independent baseline, every column sum is 0.
 * For an action-dependent baseline, column sums are generally non-zero.
 */
export function baselineExpectationMatrix(options: {
  policy: number[];
  state: number;
  baselineByAction: number[];
}): number[][] {
  const { policy, baselineByAction } = options;
  return policy.map((piA, a) => {
    const score = softmaxScore(policy, a);
    const b = baselineByAction[a] ?? 0;
    return score.map((sc) => piA * b * sc);
  });
}

export interface BaselineInvarianceResult {
  perStateComponentSum: number[][];
  maxAbs: number;
  isInvariant: boolean;
}

/**
 * Verify that a baseline leaves the policy gradient unchanged.
 *
 * For each state s and each policy preference component k we compute
 * Σ_a π(a|s) score(s,a)[k] b(s,a).  If b(s,a) is the same for every a, then
 * Σ_a π(a|s) score(s,a)[k] = 0 for every k, so the result is the zero vector
 * (up to roundoff).  If b depends on a, the sum is generally non-zero.
 */
export function checkBaselineInvariance(
  policy: Policy,
  baseline: number[][]
): BaselineInvarianceResult {
  const perStateComponentSum = policy.map((pi, s) => {
    const numActions = pi.length;
    const sum = new Array(numActions).fill(0);
    for (let a = 0; a < numActions; a++) {
      const score = softmaxScore(pi, a);
      const b = baseline[s]?.[a] ?? 0;
      for (let k = 0; k < numActions; k++) {
        sum[k] += pi[a] * score[k] * b;
      }
    }
    return sum;
  });
  const maxAbs = Math.max(
    ...perStateComponentSum.map((row) => Math.max(...row.map(Math.abs)))
  );
  return { perStateComponentSum, maxAbs, isInvariant: maxAbs < 1e-9 };
}

export function buildActionIndependentBaseline(numStates: number, value: number, numActions = 5): number[][] {
  return Array.from({ length: numStates }, () => new Array(numActions).fill(value));
}

export function buildActionDependentBaseline(q: number[][]): number[][] {
  return q.map((row) => [...row]);
}

// ---------------------------------------------------------------------------
// QAC: Q-function Actor-Critic
// ---------------------------------------------------------------------------

export function qac(config: GridWorldConfig, options: ACOptions): ACResult {
  const {
    seed,
    horizonH,
    actorAlpha,
    criticAlpha,
    episodes,
    bootstrapOnTruncation = true,
    actorWeightMode = 'raw-q',
  } = options;
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;

  let h: number[][] = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  let q: number[][] = Array.from({ length: numStates }, () => new Array(numActions).fill(0));

  const initialPolicy = h.map((row) => softmaxDistribution(row));
  let currentPolicy: Policy = initialPolicy.map((row) => [...row]);
  const policyAfterEpisode: Policy[] = [];

  const updates: ACUpdateRecord[] = [];
  const episodesRecord: ACEpisodeRecord[] = [];
  let diverged = false;
  let divergenceStep: number | undefined;
  let divergenceReason: string | undefined;
  let largeMagnitudeWarning: { step: number; reason: string } | undefined;
  let stopTraining = false;

  for (let ep = 0; ep < episodes && !stopTraining; ep++) {
    let state = config.startState;
    let cumulativeReward = 0;
    let discountedReturn = 0;
    let discount = 1;
    let episodeLength = 0;
    let success = false;
    let truncated = false;

    if (isTerminal(state, config)) {
      policyAfterEpisode.push(currentPolicy.map((row) => [...row]));
      episodesRecord.push({ cumulativeReward, discountedReturn, episodeLength, success, truncated });
      continue;
    }

    // Sarsa-style QAC: sample the first action from the current policy.
    let action = sampleActionWithRng(currentPolicy[state], rng) as Action;

    for (let t = 0; t < horizonH; t++) {
      const fullPolicyBefore = h.map((row) => softmaxDistribution(row));
      const policyBefore = fullPolicyBefore[state];
      const result = step(state, action, config);
      episodeLength = t + 1;
      cumulativeReward += result.reward;
      discountedReturn += discount * result.reward;
      discount *= config.gamma;

      const isTruncated = !result.done && t === horizonH - 1;
      const shouldBootstrap = !result.done && (!isTruncated || bootstrapOnTruncation);

      const qBefore = q[state][action];
      let criticTarget = result.reward;
      let criticBootstrap = 0;
      let nextAction: Action | undefined;
      if (shouldBootstrap) {
        const policyNext = softmaxDistribution(h[result.nextState]);
        nextAction = sampleActionWithRng(policyNext, rng) as Action;
        criticBootstrap = q[result.nextState][nextAction];
        criticTarget += config.gamma * criticBootstrap;
      }
      const qTableBefore = q.map((row) => [...row]);
      const criticError = criticTarget - qBefore;
      q[state][action] += criticAlpha * criticError;
      const qTableAfter = q.map((row) => [...row]);
      const qAfter = q[state][action];

      const score = softmaxScore(policyBefore, action);
      let actorWeight: number;
      if (actorWeightMode === 'advantage') {
        const vPi = policyBefore.reduce((sum, p, a) => sum + p * qTableBefore[state][a], 0);
        actorWeight = qBefore - vPi;
      } else {
        actorWeight = qBefore;
      }
      const actorDelta = score.map((s) => actorAlpha * actorWeight * s);
      for (let a = 0; a < numActions; a++) {
        h[state][a] += actorDelta[a];
      }
      const fullPolicyAfter = h.map((row) => softmaxDistribution(row));
      currentPolicy = fullPolicyAfter.map((row) => [...row]);

      const record: ACUpdateRecord = {
        episode: ep + 1,
        time: t,
        state,
        action,
        reward: result.reward,
        nextState: result.nextState,
        nextAction,
        done: result.done,
        truncated: isTruncated,
        bootstrapUsed: shouldBootstrap,
        actorPolicyBefore: policyBefore,
        actorPolicyAfter: fullPolicyAfter[state],
        actorFullPolicyBefore: fullPolicyBefore,
        actorFullPolicyAfter: fullPolicyAfter,
        criticEstimateBefore: qBefore,
        criticBootstrap,
        criticTarget,
        tdError: criticError,
        criticEstimateAfter: qAfter,
        actorWeight,
        actorWeightMode,
        scoreGradient: score,
        actorDelta,
        qBefore: qTableBefore,
        qAfter: qTableAfter,
      };

      const health = checkNumericalHealth(record);
      if (health.status === 'stopped' && !diverged) {
        diverged = true;
        divergenceStep = updates.length;
        divergenceReason = health.reason;
        stopTraining = true;
      } else if (health.status === 'warning' && !largeMagnitudeWarning) {
        largeMagnitudeWarning = { step: updates.length, reason: health.reason };
      }
      updates.push(record);
      if (stopTraining) break;

      if (result.done) {
        success = true;
        break;
      }
      if (isTruncated) {
        truncated = true;
        break;
      }

      state = result.nextState;
      action = nextAction!;
    }

    policyAfterEpisode.push(currentPolicy.map((row) => [...row]));
    episodesRecord.push({ cumulativeReward, discountedReturn, episodeLength, success, truncated });
  }

  const finalPolicy = h.map((row) => softmaxDistribution(row));
  return {
    updates,
    frames: updates,
    episodes: episodesRecord,
    initialPolicy,
    policyAfterEpisode,
    diverged,
    divergenceStep,
    divergenceReason,
    largeMagnitudeWarning,
    finalQ: q.map((row) => [...row]),
    finalPolicy,
  };
}

// ---------------------------------------------------------------------------
// A2C: Advantage Actor-Critic
// ---------------------------------------------------------------------------

export function a2c(config: GridWorldConfig, options: ACOptions): ACResult {
  const { seed, horizonH, actorAlpha, criticAlpha, episodes, bootstrapOnTruncation = true } = options;
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;

  let h: number[][] = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  let v: number[] = new Array(numStates).fill(0);

  const initialPolicy = h.map((row) => softmaxDistribution(row));
  let currentPolicy: Policy = initialPolicy.map((row) => [...row]);
  const policyAfterEpisode: Policy[] = [];

  const updates: ACUpdateRecord[] = [];
  const episodesRecord: ACEpisodeRecord[] = [];
  let diverged = false;
  let divergenceStep: number | undefined;
  let divergenceReason: string | undefined;
  let largeMagnitudeWarning: { step: number; reason: string } | undefined;
  let stopTraining = false;

  for (let ep = 0; ep < episodes && !stopTraining; ep++) {
    const episodeStartUpdateCount = updates.length;
    let state = config.startState;
    let cumulativeReward = 0;
    let discountedReturn = 0;
    let discount = 1;
    let episodeLength = 0;
    let success = false;
    let truncated = false;

    for (let t = 0; t < horizonH; t++) {
      if (isTerminal(state, config)) break;

      const fullPolicyBefore = h.map((row) => softmaxDistribution(row));
      const policyBefore = fullPolicyBefore[state];
      const action = sampleActionWithRng(policyBefore, rng);
      const result = step(state, action as Action, config);
      episodeLength = t + 1;
      cumulativeReward += result.reward;
      discountedReturn += discount * result.reward;
      discount *= config.gamma;

      const isTruncated = !result.done && t === horizonH - 1;
      const shouldBootstrap = !result.done && (!isTruncated || bootstrapOnTruncation);

      const vBeforeAll = [...v];
      const vBefore = v[state];
      const criticBootstrap = shouldBootstrap ? v[result.nextState] : 0;
      const criticTarget = result.reward + config.gamma * criticBootstrap;
      const tdError = criticTarget - vBefore;
      v[state] += criticAlpha * tdError;
      const vAfterAll = [...v];

      const score = softmaxScore(policyBefore, action);
      const actorWeight = tdError;
      const actorDelta = score.map((s) => actorAlpha * actorWeight * s);
      for (let a = 0; a < numActions; a++) {
        h[state][a] += actorDelta[a];
      }
      const fullPolicyAfter = h.map((row) => softmaxDistribution(row));
      currentPolicy = fullPolicyAfter.map((row) => [...row]);

      const record: ACUpdateRecord = {
        episode: ep + 1,
        time: t,
        state,
        action: action as Action,
        reward: result.reward,
        nextState: result.nextState,
        done: result.done,
        truncated: isTruncated,
        bootstrapUsed: shouldBootstrap,
        actorPolicyBefore: policyBefore,
        actorPolicyAfter: fullPolicyAfter[state],
        actorFullPolicyBefore: fullPolicyBefore,
        actorFullPolicyAfter: fullPolicyAfter,
        criticEstimateBefore: vBefore,
        criticBootstrap,
        criticTarget,
        tdError,
        criticEstimateAfter: vAfterAll[state],
        actorWeight,
        scoreGradient: score,
        actorDelta,
        vBefore: vBeforeAll,
        vAfter: vAfterAll,
      };

      const health = checkNumericalHealth(record);
      if (health.status === 'stopped' && !diverged) {
        diverged = true;
        divergenceStep = updates.length;
        divergenceReason = health.reason;
        stopTraining = true;
      } else if (health.status === 'warning' && !largeMagnitudeWarning) {
        largeMagnitudeWarning = { step: updates.length, reason: health.reason };
      }
      updates.push(record);
      if (stopTraining) break;

      state = result.nextState;
      if (result.done) {
        success = true;
        break;
      }
      if (isTruncated) {
        truncated = true;
        break;
      }
    }

    if (updates.length > episodeStartUpdateCount) {
      currentPolicy = updates[updates.length - 1].actorFullPolicyAfter!;
    }
    policyAfterEpisode.push(currentPolicy.map((row) => [...row]));
    episodesRecord.push({ cumulativeReward, discountedReturn, episodeLength, success, truncated });
  }

  const finalPolicy = h.map((row) => softmaxDistribution(row));
  return {
    updates,
    frames: updates,
    episodes: episodesRecord,
    initialPolicy,
    policyAfterEpisode,
    diverged,
    divergenceStep,
    divergenceReason,
    largeMagnitudeWarning,
    finalV: [...v],
    finalPolicy,
  };
}

// ---------------------------------------------------------------------------
// Off-policy Actor-Critic (V-based, textbook Algorithm 10.3 style)
// ---------------------------------------------------------------------------

export function offPolicyActorCritic(config: GridWorldConfig, options: ACOptions): ACResult {
  const {
    seed,
    horizonH,
    actorAlpha,
    criticAlpha,
    episodes,
    epsilon = 0.5,
    bootstrapOnTruncation = true,
    importanceMode = 'raw',
    clipThreshold = 10,
  } = options;
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;

  let h: number[][] = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  let v: number[] = new Array(numStates).fill(0);

  const initialPolicy = h.map((row) => softmaxDistribution(row));
  let currentPolicy: Policy = initialPolicy.map((row) => [...row]);
  const policyAfterEpisode: Policy[] = [];

  const updates: ACUpdateRecord[] = [];
  const episodesRecord: ACEpisodeRecord[] = [];
  let diverged = false;
  let divergenceStep: number | undefined;
  let divergenceReason: string | undefined;
  let largeMagnitudeWarning: { step: number; reason: string } | undefined;
  let stopTraining = false;

  function clipRho(raw: number): { used: number; clipped: boolean } {
    if (importanceMode !== 'clipped') return { used: raw, clipped: false };
    const threshold = Math.max(1, clipThreshold);
    return raw > threshold ? { used: threshold, clipped: true } : { used: raw, clipped: false };
  }

  for (let ep = 0; ep < episodes && !stopTraining; ep++) {
    const episodeStartUpdateCount = updates.length;
    let state = config.startState;
    let cumulativeReward = 0;
    let discountedReturn = 0;
    let discount = 1;
    let episodeLength = 0;
    let success = false;
    let truncated = false;

    for (let t = 0; t < horizonH; t++) {
      if (isTerminal(state, config)) break;

      const fullPolicyBefore = h.map((row) => softmaxDistribution(row));
      const targetPolicy = fullPolicyBefore[state];
      const behaviorPolicy = targetPolicy.map((p) => epsilon / numActions + (1 - epsilon) * p);
      checkCoverage(targetPolicy, behaviorPolicy);

      const action = sampleActionWithRng(behaviorPolicy, rng);
      const result = step(state, action as Action, config);
      episodeLength = t + 1;
      cumulativeReward += result.reward;
      discountedReturn += discount * result.reward;
      discount *= config.gamma;

      const targetProb = targetPolicy[action];
      const behaviorProb = behaviorPolicy[action];
      const rawRho = targetProb / behaviorProb;
      const { used: usedRho, clipped: wasClipped } = clipRho(rawRho);

      const isTruncated = !result.done && t === horizonH - 1;
      const shouldBootstrap = !result.done && (!isTruncated || bootstrapOnTruncation);

      const vBeforeAll = [...v];
      const vBefore = v[state];
      const criticBootstrap = shouldBootstrap ? v[result.nextState] : 0;
      const criticTarget = result.reward + config.gamma * criticBootstrap;
      const tdError = criticTarget - vBefore;
      v[state] += criticAlpha * usedRho * tdError;
      const vAfterAll = [...v];

      const score = softmaxScore(targetPolicy, action);
      const actorWeight = usedRho * tdError;
      const actorDelta = score.map((s) => actorAlpha * actorWeight * s);
      for (let a = 0; a < numActions; a++) {
        h[state][a] += actorDelta[a];
      }
      const fullPolicyAfter = h.map((row) => softmaxDistribution(row));
      currentPolicy = fullPolicyAfter.map((row) => [...row]);

      const record: ACUpdateRecord = {
        episode: ep + 1,
        time: t,
        state,
        action: action as Action,
        reward: result.reward,
        nextState: result.nextState,
        done: result.done,
        truncated: isTruncated,
        bootstrapUsed: shouldBootstrap,
        actorPolicyBefore: targetPolicy,
        actorPolicyAfter: fullPolicyAfter[state],
        actorFullPolicyBefore: fullPolicyBefore,
        actorFullPolicyAfter: fullPolicyAfter,
        criticEstimateBefore: vBefore,
        criticBootstrap,
        criticTarget,
        tdError,
        criticEstimateAfter: vAfterAll[state],
        actorWeight,
        scoreGradient: score,
        actorDelta,
        vBefore: vBeforeAll,
        vAfter: vAfterAll,
        targetPolicy,
        behaviorPolicy,
        targetProb,
        behaviorProb,
        rho: usedRho,
        rawRho,
        usedRho,
        wasClipped,
      };

      const health = checkNumericalHealth(record);
      if (health.status === 'stopped' && !diverged) {
        diverged = true;
        divergenceStep = updates.length;
        divergenceReason = health.reason;
        stopTraining = true;
      } else if (health.status === 'warning' && !largeMagnitudeWarning) {
        largeMagnitudeWarning = { step: updates.length, reason: health.reason };
      }
      updates.push(record);
      if (stopTraining) break;

      state = result.nextState;
      if (result.done) {
        success = true;
        break;
      }
      if (isTruncated) {
        truncated = true;
        break;
      }
    }

    if (updates.length > episodeStartUpdateCount) {
      currentPolicy = updates[updates.length - 1].actorFullPolicyAfter!;
    }
    policyAfterEpisode.push(currentPolicy.map((row) => [...row]));
    episodesRecord.push({ cumulativeReward, discountedReturn, episodeLength, success, truncated });
  }

  const finalPolicy = h.map((row) => softmaxDistribution(row));
  return {
    updates,
    frames: updates,
    episodes: episodesRecord,
    initialPolicy,
    policyAfterEpisode,
    diverged,
    divergenceStep,
    divergenceReason,
    largeMagnitudeWarning,
    finalV: [...v],
    finalPolicy,
  };
}

// ---------------------------------------------------------------------------
// Q-based off-policy Actor-Critic
// ---------------------------------------------------------------------------

export function qBasedOffPolicyActorCritic(config: GridWorldConfig, options: ACOptions): ACResult {
  const {
    seed,
    horizonH,
    actorAlpha,
    criticAlpha,
    episodes,
    epsilon = 0.5,
    bootstrapOnTruncation = true,
    importanceMode = 'raw',
    clipThreshold = 10,
  } = options;
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;

  let h: number[][] = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  let q: number[][] = Array.from({ length: numStates }, () => new Array(numActions).fill(0));

  const initialPolicy = h.map((row) => softmaxDistribution(row));
  let currentPolicy: Policy = initialPolicy.map((row) => [...row]);
  const policyAfterEpisode: Policy[] = [];

  const updates: ACUpdateRecord[] = [];
  const episodesRecord: ACEpisodeRecord[] = [];
  let diverged = false;
  let divergenceStep: number | undefined;
  let divergenceReason: string | undefined;
  let largeMagnitudeWarning: { step: number; reason: string } | undefined;
  let stopTraining = false;

  function behaviorPolicyFromQ(state: number): number[] {
    const qState = q[state];
    const maxQ = Math.max(...qState);
    const best = qState.map((val) => (Math.abs(val - maxQ) < 1e-9 ? 1 : 0)) as number[];
    const numBest = best.reduce((a, b) => a + Number(b), 0);
    const greedy = best.map((isBest) => (isBest ? 1 / numBest : 0));
    return greedy.map((p) => epsilon / numActions + (1 - epsilon) * p);
  }

  function clipRho(raw: number): { used: number; clipped: boolean } {
    if (importanceMode !== 'clipped') return { used: raw, clipped: false };
    const threshold = Math.max(1, clipThreshold);
    return raw > threshold ? { used: threshold, clipped: true } : { used: raw, clipped: false };
  }

  for (let ep = 0; ep < episodes && !stopTraining; ep++) {
    const episodeStartUpdateCount = updates.length;
    let state = config.startState;
    let cumulativeReward = 0;
    let discountedReturn = 0;
    let discount = 1;
    let episodeLength = 0;
    let success = false;
    let truncated = false;

    for (let t = 0; t < horizonH; t++) {
      if (isTerminal(state, config)) break;

      const fullPolicyBefore = h.map((row) => softmaxDistribution(row));
      const targetPolicy = fullPolicyBefore[state];
      const behaviorPolicy = behaviorPolicyFromQ(state);
      checkCoverage(targetPolicy, behaviorPolicy);

      const action = sampleActionWithRng(behaviorPolicy, rng);
      const result = step(state, action as Action, config);
      episodeLength = t + 1;
      cumulativeReward += result.reward;
      discountedReturn += discount * result.reward;
      discount *= config.gamma;

      const targetProb = targetPolicy[action];
      const behaviorProb = behaviorPolicy[action];
      const rawRho = targetProb / behaviorProb;
      const { used: usedRho, clipped: wasClipped } = clipRho(rawRho);

      const isTruncated = !result.done && t === horizonH - 1;
      const shouldBootstrap = !result.done && (!isTruncated || bootstrapOnTruncation);

      const qBefore = q[state][action];
      let criticBootstrap = 0;
      let expectedBootstrap = 0;
      let criticTarget = result.reward;
      if (shouldBootstrap) {
        const targetPolicyNext = softmaxDistribution(h[result.nextState]);
        expectedBootstrap = targetPolicyNext.reduce((sum, p, a) => sum + p * q[result.nextState][a], 0);
        criticBootstrap = expectedBootstrap;
        criticTarget += config.gamma * expectedBootstrap;
      }
      const qTableBefore = q.map((row) => [...row]);
      const criticError = criticTarget - qBefore;
      q[state][action] += criticAlpha * usedRho * criticError;
      const qTableAfter = q.map((row) => [...row]);
      const qAfter = q[state][action];

      const score = softmaxScore(targetPolicy, action);
      const actorWeight = usedRho * qBefore;
      const actorDelta = score.map((s) => actorAlpha * actorWeight * s);
      for (let a = 0; a < numActions; a++) {
        h[state][a] += actorDelta[a];
      }
      const fullPolicyAfter = h.map((row) => softmaxDistribution(row));
      currentPolicy = fullPolicyAfter.map((row) => [...row]);

      const record: ACUpdateRecord = {
        episode: ep + 1,
        time: t,
        state,
        action: action as Action,
        reward: result.reward,
        nextState: result.nextState,
        done: result.done,
        truncated: isTruncated,
        bootstrapUsed: shouldBootstrap,
        actorPolicyBefore: targetPolicy,
        actorPolicyAfter: fullPolicyAfter[state],
        actorFullPolicyBefore: fullPolicyBefore,
        actorFullPolicyAfter: fullPolicyAfter,
        criticEstimateBefore: qBefore,
        criticBootstrap,
        expectedBootstrap,
        criticTarget,
        tdError: criticError,
        criticEstimateAfter: qAfter,
        actorWeight,
        scoreGradient: score,
        actorDelta,
        qBefore: qTableBefore,
        qAfter: qTableAfter,
        targetPolicy,
        behaviorPolicy,
        targetProb,
        behaviorProb,
        rho: usedRho,
        rawRho,
        usedRho,
        wasClipped,
      };

      const health = checkNumericalHealth(record);
      if (health.status === 'stopped' && !diverged) {
        diverged = true;
        divergenceStep = updates.length;
        divergenceReason = health.reason;
        stopTraining = true;
      } else if (health.status === 'warning' && !largeMagnitudeWarning) {
        largeMagnitudeWarning = { step: updates.length, reason: health.reason };
      }
      updates.push(record);
      if (stopTraining) break;

      state = result.nextState;
      if (result.done) {
        success = true;
        break;
      }
      if (isTruncated) {
        truncated = true;
        break;
      }
    }

    if (updates.length > episodeStartUpdateCount) {
      currentPolicy = updates[updates.length - 1].actorFullPolicyAfter!;
    }
    policyAfterEpisode.push(currentPolicy.map((row) => [...row]));
    episodesRecord.push({ cumulativeReward, discountedReturn, episodeLength, success, truncated });
  }

  const finalPolicy = h.map((row) => softmaxDistribution(row));
  return {
    updates,
    frames: updates,
    episodes: episodesRecord,
    initialPolicy,
    policyAfterEpisode,
    diverged,
    divergenceStep,
    divergenceReason,
    largeMagnitudeWarning,
    finalQ: q.map((row) => [...row]),
    finalPolicy,
  };
}

// ---------------------------------------------------------------------------
// Deterministic policy gradient toy
// ---------------------------------------------------------------------------

export interface DeterministicACStep {
  step: number;
  state: number;
  actorAction: number;
  behaviorAction: number;
  explorationNoise: number;
  reward: number;
  nextState: number;
  done: boolean;
  nextActorAction?: number;
  target: number;
  prediction: number;
  tdError: number;
  criticLoss: number;
  dqda: number;
  dmuDtheta: number[];
  actorGradient: number[];
  wBefore: number[];
  wAfter: number[];
  thetaBefore: number[];
  thetaAfter: number[];
  criticWeightsUsedByActor: 'before' | 'after';
}

export interface DeterministicACResult {
  steps: DeterministicACStep[];
  finalW: number[];
  finalTheta: number[];
  diverged: boolean;
  divergenceStep?: number;
  divergenceReason?: string;
  largeMagnitudeWarning?: { step: number; reason: string };
}

export interface DeterministicACEnv {
  step: (state: number, action: number) => { nextState: number; reward: number; done: boolean };
  terminal: (state: number) => boolean;
}

function checkDeterministicStability(
  tdError: number,
  dqda: number,
  w: number[],
  theta: number[],
  actorGradient: number[]
): NumericalHealth {
  if (!finite(tdError) || !finite(dqda) || !allFinite(actorGradient) || !allFinite(w) || !allFinite(theta)) {
    return { status: 'stopped', reason: 'non-finite tdError, dqda, gradient, w or theta' };
  }
  const reasons: string[] = [];
  if (Math.abs(tdError) > SAFE_THRESHOLDS.tdError) reasons.push(`|tdError|=${tdError.toExponential(2)}`);
  if (Math.abs(dqda) > SAFE_THRESHOLDS.actorGradientNorm) reasons.push(`|dqda|=${dqda.toExponential(2)}`);
  const wMax = Math.max(...w.map(Math.abs));
  const thetaMax = Math.max(...theta.map(Math.abs));
  const gradMax = Math.max(...actorGradient.map(Math.abs));
  if (wMax > SAFE_THRESHOLDS.criticEstimate) reasons.push(`|w|_max=${wMax.toExponential(2)}`);
  if (thetaMax > SAFE_THRESHOLDS.actorWeight) reasons.push(`|theta|_max=${thetaMax.toExponential(2)}`);
  if (gradMax > SAFE_THRESHOLDS.actorGradientNorm) reasons.push(`|actor grad|_max=${gradMax.toExponential(2)}`);
  if (reasons.length > 0) return { status: 'warning', reason: reasons.join('; ') };
  return { status: 'stable' };
}

export function deterministicActorCriticEpisode(
  env: DeterministicACEnv,
  initialState: number,
  w: number[],
  theta: number[],
  options: {
    seed: number;
    horizonH: number;
    actorAlpha: number;
    criticAlpha: number;
    gamma: number;
    qValueWithWeights: (s: number, a: number, w: number[]) => number;
    muWithTheta: (s: number, theta: number[]) => number;
    gradMuWithTheta: (s: number, theta: number[]) => number[];
    gradQWrtA: (s: number, a: number, w: number[]) => number;
    criticFeatures: (s: number, a: number) => number[];
    explorationNoiseStd?: number;
    criticWeightsUsedByActor?: 'before' | 'after';
  }
): DeterministicACResult {
  const {
    seed,
    horizonH,
    actorAlpha,
    criticAlpha,
    gamma,
    qValueWithWeights,
    muWithTheta,
    gradMuWithTheta,
    gradQWrtA,
    criticFeatures,
    explorationNoiseStd = 0,
    criticWeightsUsedByActor = 'before',
  } = options;
  const rng = mulberry32(seed);
  const steps: DeterministicACStep[] = [];

  let currentW = [...w];
  let currentTheta = [...theta];
  let state = initialState;
  let diverged = false;
  let divergenceStep: number | undefined;
  let divergenceReason: string | undefined;
  let largeMagnitudeWarning: { step: number; reason: string } | undefined;

  for (let t = 0; t < horizonH; t++) {
    const actorAction = muWithTheta(state, currentTheta);
    let noise = 0;
    if (explorationNoiseStd > 0) {
      const u1 = rng();
      const u2 = rng();
      noise = Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
    }
    const behaviorAction = actorAction + explorationNoiseStd * noise;

    const result = env.step(state, behaviorAction);
    const nextActorAction = result.done ? undefined : muWithTheta(result.nextState, currentTheta);

    const target = result.done
      ? result.reward
      : result.reward + gamma * qValueWithWeights(result.nextState, nextActorAction!, currentW);
    const prediction = qValueWithWeights(state, behaviorAction, currentW);
    const tdError = target - prediction;
    const criticLoss = 0.5 * tdError * tdError;

    const wBefore = [...currentW];
    const thetaBefore = [...currentTheta];

    // Critic update uses the behavior action actually executed.
    const phi = criticFeatures(state, behaviorAction);
    for (let i = 0; i < currentW.length; i++) {
      currentW[i] += criticAlpha * tdError * phi[i];
    }

    // Actor gradient is computed at the deterministic actor action.
    // By default use wBefore so the Actor signal and Critic update are clearly timed.
    const actorCriticWeights = criticWeightsUsedByActor === 'before' ? wBefore : currentW;
    const dqda = gradQWrtA(state, actorAction, actorCriticWeights);
    const dmuDtheta = gradMuWithTheta(state, currentTheta);
    const actorGradient = dmuDtheta.map((g) => actorAlpha * dqda * g);
    for (let i = 0; i < currentTheta.length; i++) {
      currentTheta[i] += actorGradient[i];
    }

    const stepRecord: DeterministicACStep = {
      step: t,
      state,
      actorAction,
      behaviorAction,
      explorationNoise: explorationNoiseStd * noise,
      reward: result.reward,
      nextState: result.nextState,
      done: result.done,
      nextActorAction,
      target,
      prediction,
      tdError,
      criticLoss,
      dqda,
      dmuDtheta,
      actorGradient,
      wBefore,
      wAfter: [...currentW],
      thetaBefore,
      thetaAfter: [...currentTheta],
      criticWeightsUsedByActor,
    };

    const health = checkDeterministicStability(tdError, dqda, currentW, currentTheta, actorGradient);
    if (health.status === 'stopped' && !diverged) {
      diverged = true;
      divergenceStep = t;
      divergenceReason = health.reason;
      steps.push(stepRecord);
      break;
    } else if (health.status === 'warning' && !largeMagnitudeWarning) {
      largeMagnitudeWarning = { step: t, reason: health.reason };
    }
    steps.push(stepRecord);

    state = result.nextState;
    if (result.done || env.terminal(state)) break;
  }

  return { steps, finalW: currentW, finalTheta: currentTheta, diverged, divergenceStep, divergenceReason, largeMagnitudeWarning };
}
