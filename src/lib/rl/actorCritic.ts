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
} from './gridworld';

export type ACFrame = ACUpdateRecord;

export interface ACUpdateRecord {
  episode: number;
  time: number;

  state: number;
  action: Action;
  reward: number;
  nextState: number;

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
  diverged: boolean;
  divergenceStep?: number;
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

function checkDivergence(record: ACUpdateRecord): boolean {
  if (!finite(record.tdError) || !finite(record.actorWeight)) return true;
  if (!allFinite(record.actorDelta) || !allFinite(record.scoreGradient)) return true;
  if (!allFinite(record.actorPolicyBefore) || !allFinite(record.actorPolicyAfter)) return true;
  if (record.actorFullPolicyBefore && !record.actorFullPolicyBefore.every((dist) => allFinite(dist))) return true;
  if (record.actorFullPolicyAfter && !record.actorFullPolicyAfter.every((dist) => allFinite(dist))) return true;
  if (record.vBefore && !allFinite(record.vBefore)) return true;
  if (record.vAfter && !allFinite(record.vAfter)) return true;
  if (record.qBefore && !record.qBefore.every((row) => allFinite(row))) return true;
  if (record.qAfter && !record.qAfter.every((row) => allFinite(row))) return true;
  return false;
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

export function klDivergence(pi: number[], old: number[]): number {
  return pi.reduce((sum, p, a) => (p > 0 && old[a] > 0 ? sum + p * Math.log(p / old[a]) : sum), 0);
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
  returnValue: number;
  movingAverage: number;
  length: number;
  success: number;
  tdError: number;
  actorGradientNorm: number;
  criticUpdateNorm: number;
  entropy: number;
  kl: number;
}

export function computeACMetricSeries(
  result: ACResult,
  policyHistory: Policy[],
  window = 10
): ACMetricSeries[] {
  const perEpisode: Record<number, { returns: number[]; lengths: number[]; tdErrors: number[]; actorNorms: number[]; criticNorms: number[]; entropies: number[]; kls: number[] }> = {};
  result.updates.forEach((u) => {
    const e = u.episode;
    if (!perEpisode[e]) {
      perEpisode[e] = { returns: [], lengths: [], tdErrors: [], actorNorms: [], criticNorms: [], entropies: [], kls: [] };
    }
    perEpisode[e].tdErrors.push(u.tdError);
    perEpisode[e].actorNorms.push(vectorL2(u.actorDelta));
    const critUpdate = u.vAfter
      ? Math.abs(u.criticEstimateAfter - u.criticEstimateBefore)
      : 0;
    perEpisode[e].criticNorms.push(critUpdate);
  });

  result.episodes.forEach((ep, i) => {
    const e = i + 1;
    if (!perEpisode[e]) perEpisode[e] = { returns: [], lengths: [], tdErrors: [], actorNorms: [], criticNorms: [], entropies: [], kls: [] };
    perEpisode[e].returns.push(ep.cumulativeReward);
    perEpisode[e].lengths.push(ep.episodeLength);
  });

  policyHistory.forEach((pol, i) => {
    const e = i;
    if (!perEpisode[e]) perEpisode[e] = { returns: [], lengths: [], tdErrors: [], actorNorms: [], criticNorms: [], entropies: [], kls: [] };
    const ent = pol.reduce((sum, dist) => sum + computeEntropy(dist), 0) / pol.length;
    perEpisode[e].entropies.push(ent);
    if (i > 0) {
      const kl = pol.reduce((sum, dist, s) => sum + klDivergence(dist, policyHistory[i - 1][s]), 0);
      perEpisode[e].kls.push(kl);
    }
  });

  const episodes = Object.keys(perEpisode)
    .map(Number)
    .sort((a, b) => a - b);
  const returns = episodes.map((e) => perEpisode[e].returns[0] ?? 0);
  const ma = movingAverage(returns, window);

  return episodes.map((e, i) => ({
    episode: e,
    returnValue: returns[i],
    movingAverage: ma[i],
    length: perEpisode[e].lengths[0] ?? 0,
    success: perEpisode[e].returns[0] > 0 ? 1 : 0,
    tdError: average(perEpisode[e].tdErrors),
    actorGradientNorm: average(perEpisode[e].actorNorms),
    criticUpdateNorm: average(perEpisode[e].criticNorms),
    entropy: average(perEpisode[e].entropies),
    kl: average(perEpisode[e].kls),
  }));
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

// ---------------------------------------------------------------------------
// QAC: Q-function Actor-Critic
// ---------------------------------------------------------------------------

export function qac(config: GridWorldConfig, options: ACOptions): ACResult {
  const { seed, horizonH, actorAlpha, criticAlpha, episodes } = options;
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;

  let h: number[][] = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  let q: number[][] = Array.from({ length: numStates }, () => new Array(numActions).fill(0));

  const updates: ACUpdateRecord[] = [];
  const episodesRecord: ACEpisodeRecord[] = [];
  let diverged = false;
  let divergenceStep: number | undefined;

  for (let ep = 0; ep < episodes; ep++) {
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

      const qBefore = q[state][action];
      let criticTarget = result.reward;
      let criticBootstrap = 0;
      if (!result.done) {
        const policyNext = softmaxDistribution(h[result.nextState]);
        const aNext = sampleActionWithRng(policyNext, rng);
        criticBootstrap = q[result.nextState][aNext];
        criticTarget += config.gamma * criticBootstrap;
      }
      const qTableBefore = q.map((row) => [...row]);
      const criticError = criticTarget - qBefore;
      q[state][action] += criticAlpha * criticError;
      const qTableAfter = q.map((row) => [...row]);
      const qAfter = q[state][action];

      const score = softmaxScore(policyBefore, action);
      const actorWeight = qBefore;
      const actorDelta = score.map((s) => actorAlpha * actorWeight * s);
      for (let a = 0; a < numActions; a++) {
        h[state][a] += actorDelta[a];
      }
      const fullPolicyAfter = h.map((row) => softmaxDistribution(row));

      const record: ACUpdateRecord = {
        episode: ep + 1,
        time: t,
        state,
        action: action as Action,
        reward: result.reward,
        nextState: result.nextState,
        done: result.done,
        truncated: false,
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
        scoreGradient: score,
        actorDelta,
        qBefore: qTableBefore,
        qAfter: qTableAfter,
      };

      if (!diverged && checkDivergence(record)) {
        diverged = true;
        divergenceStep = updates.length;
      }
      updates.push(record);

      state = result.nextState;
      if (result.done) {
        success = true;
        break;
      }
      if (t === horizonH - 1) {
        truncated = true;
      }
    }

    episodesRecord.push({ cumulativeReward, discountedReturn, episodeLength, success, truncated });
  }

  return {
    updates,
    frames: updates,
    episodes: episodesRecord,
    diverged,
    divergenceStep,
    finalQ: q.map((row) => [...row]),
    finalPolicy: h.map((row) => softmaxDistribution(row)),
  };
}

// ---------------------------------------------------------------------------
// A2C: Advantage Actor-Critic
// ---------------------------------------------------------------------------

export function a2c(config: GridWorldConfig, options: ACOptions): ACResult {
  const { seed, horizonH, actorAlpha, criticAlpha, episodes } = options;
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;

  let h: number[][] = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  let v: number[] = new Array(numStates).fill(0);

  const updates: ACUpdateRecord[] = [];
  const episodesRecord: ACEpisodeRecord[] = [];
  let diverged = false;
  let divergenceStep: number | undefined;

  for (let ep = 0; ep < episodes; ep++) {
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

      const vBeforeAll = [...v];
      const vBefore = v[state];
      const criticBootstrap = result.done ? 0 : v[result.nextState];
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

      const record: ACUpdateRecord = {
        episode: ep + 1,
        time: t,
        state,
        action: action as Action,
        reward: result.reward,
        nextState: result.nextState,
        done: result.done,
        truncated: false,
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

      if (!diverged && checkDivergence(record)) {
        diverged = true;
        divergenceStep = updates.length;
      }
      updates.push(record);

      state = result.nextState;
      if (result.done) {
        success = true;
        break;
      }
      if (t === horizonH - 1) truncated = true;
    }

    episodesRecord.push({ cumulativeReward, discountedReturn, episodeLength, success, truncated });
  }

  return {
    updates,
    frames: updates,
    episodes: episodesRecord,
    diverged,
    divergenceStep,
    finalV: [...v],
    finalPolicy: h.map((row) => softmaxDistribution(row)),
  };
}

// ---------------------------------------------------------------------------
// Off-policy Actor-Critic (V-based, textbook Algorithm 10.3 style)
// ---------------------------------------------------------------------------

export function offPolicyActorCritic(config: GridWorldConfig, options: ACOptions): ACResult {
  const { seed, horizonH, actorAlpha, criticAlpha, episodes, epsilon = 0.5 } = options;
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;

  let h: number[][] = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  let v: number[] = new Array(numStates).fill(0);

  const updates: ACUpdateRecord[] = [];
  const episodesRecord: ACEpisodeRecord[] = [];
  let diverged = false;
  let divergenceStep: number | undefined;

  for (let ep = 0; ep < episodes; ep++) {
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
      const rho = targetProb / behaviorProb;

      const vBeforeAll = [...v];
      const vBefore = v[state];
      const criticBootstrap = result.done ? 0 : v[result.nextState];
      const criticTarget = result.reward + config.gamma * criticBootstrap;
      const tdError = criticTarget - vBefore;
      v[state] += criticAlpha * rho * tdError;
      const vAfterAll = [...v];

      const score = softmaxScore(targetPolicy, action);
      const actorWeight = rho * tdError;
      const actorDelta = score.map((s) => actorAlpha * actorWeight * s);
      for (let a = 0; a < numActions; a++) {
        h[state][a] += actorDelta[a];
      }
      const fullPolicyAfter = h.map((row) => softmaxDistribution(row));

      const record: ACUpdateRecord = {
        episode: ep + 1,
        time: t,
        state,
        action: action as Action,
        reward: result.reward,
        nextState: result.nextState,
        done: result.done,
        truncated: false,
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
        rho,
      };

      if (!diverged && checkDivergence(record)) {
        diverged = true;
        divergenceStep = updates.length;
      }
      updates.push(record);

      state = result.nextState;
      if (result.done) {
        success = true;
        break;
      }
      if (t === horizonH - 1) truncated = true;
    }

    episodesRecord.push({ cumulativeReward, discountedReturn, episodeLength, success, truncated });
  }

  return {
    updates,
    frames: updates,
    episodes: episodesRecord,
    diverged,
    divergenceStep,
    finalV: [...v],
    finalPolicy: h.map((row) => softmaxDistribution(row)),
  };
}

// ---------------------------------------------------------------------------
// Q-based off-policy Actor-Critic
// ---------------------------------------------------------------------------

export function qBasedOffPolicyActorCritic(config: GridWorldConfig, options: ACOptions): ACResult {
  const { seed, horizonH, actorAlpha, criticAlpha, episodes, epsilon = 0.5 } = options;
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;

  let h: number[][] = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  let q: number[][] = Array.from({ length: numStates }, () => new Array(numActions).fill(0));

  const updates: ACUpdateRecord[] = [];
  const episodesRecord: ACEpisodeRecord[] = [];
  let diverged = false;
  let divergenceStep: number | undefined;

  function behaviorPolicyFromQ(state: number): number[] {
    const qState = q[state];
    const maxQ = Math.max(...qState);
    const best = qState.map((val) => (Math.abs(val - maxQ) < 1e-9 ? 1 : 0)) as number[];
    const numBest = best.reduce((a, b) => a + Number(b), 0);
    const greedy = best.map((isBest) => (isBest ? 1 / numBest : 0));
    return greedy.map((p) => epsilon / numActions + (1 - epsilon) * p);
  }

  for (let ep = 0; ep < episodes; ep++) {
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
      const rho = targetProb / behaviorProb;

      const qBefore = q[state][action];
      let criticBootstrap = 0;
      let criticTarget = result.reward;
      if (!result.done) {
        const policyNext = softmaxDistribution(h[result.nextState]);
        const aNext = sampleActionWithRng(policyNext, rng);
        criticBootstrap = q[result.nextState][aNext];
        criticTarget += config.gamma * criticBootstrap;
      }
      const qTableBefore = q.map((row) => [...row]);
      const criticError = criticTarget - qBefore;
      q[state][action] += criticAlpha * rho * criticError;
      const qTableAfter = q.map((row) => [...row]);
      const qAfter = q[state][action];

      const score = softmaxScore(targetPolicy, action);
      const actorWeight = rho * qBefore;
      const actorDelta = score.map((s) => actorAlpha * actorWeight * s);
      for (let a = 0; a < numActions; a++) {
        h[state][a] += actorDelta[a];
      }
      const fullPolicyAfter = h.map((row) => softmaxDistribution(row));

      const record: ACUpdateRecord = {
        episode: ep + 1,
        time: t,
        state,
        action: action as Action,
        reward: result.reward,
        nextState: result.nextState,
        done: result.done,
        truncated: false,
        actorPolicyBefore: targetPolicy,
        actorPolicyAfter: fullPolicyAfter[state],
        actorFullPolicyBefore: fullPolicyBefore,
        actorFullPolicyAfter: fullPolicyAfter,
        criticEstimateBefore: qBefore,
        criticBootstrap,
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
        rho,
      };

      if (!diverged && checkDivergence(record)) {
        diverged = true;
        divergenceStep = updates.length;
      }
      updates.push(record);

      state = result.nextState;
      if (result.done) {
        success = true;
        break;
      }
      if (t === horizonH - 1) truncated = true;
    }

    episodesRecord.push({ cumulativeReward, discountedReturn, episodeLength, success, truncated });
  }

  return {
    updates,
    frames: updates,
    episodes: episodesRecord,
    diverged,
    divergenceStep,
    finalQ: q.map((row) => [...row]),
    finalPolicy: h.map((row) => softmaxDistribution(row)),
  };
}

// ---------------------------------------------------------------------------
// Deterministic policy gradient toy
// ---------------------------------------------------------------------------

export interface DeterministicACStep {
  step: number;
  state: number;
  action: number;
  reward: number;
  nextState: number;
  done: boolean;
  nextAction?: number;
  target: number;
  prediction: number;
  tdError: number;
  dqda: number;
  dmuDtheta: number[];
  actorGradient: number[];
  wBefore: number[];
  wAfter: number[];
  thetaBefore: number[];
  thetaAfter: number[];
}

export interface DeterministicACResult {
  steps: DeterministicACStep[];
  finalW: number[];
  finalTheta: number[];
  diverged: boolean;
  divergenceStep?: number;
}

export interface DeterministicACEnv {
  step: (state: number, action: number) => { nextState: number; reward: number; done: boolean };
  terminal: (state: number) => boolean;
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
  } = options;
  const rng = mulberry32(seed);
  const steps: DeterministicACStep[] = [];

  let currentW = [...w];
  let currentTheta = [...theta];
  let state = initialState;
  let diverged = false;
  let divergenceStep: number | undefined;

  for (let t = 0; t < horizonH; t++) {
    let action = muWithTheta(state, currentTheta);
    if (explorationNoiseStd > 0) {
      // Box-Muller style simple normal approx via uniform
      const u1 = rng();
      const u2 = rng();
      const z = Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
      action += explorationNoiseStd * z;
    }

    const result = env.step(state, action);
    const nextAction = result.done ? undefined : muWithTheta(result.nextState, currentTheta);

    const target = result.done
      ? result.reward
      : result.reward + gamma * qValueWithWeights(result.nextState, nextAction!, currentW);
    const prediction = qValueWithWeights(state, action, currentW);
    const tdError = target - prediction;

    const wBefore = [...currentW];
    const thetaBefore = [...currentTheta];

    // Critic update
    const phi = criticFeatures(state, action);
    for (let i = 0; i < currentW.length; i++) {
      currentW[i] += criticAlpha * tdError * phi[i];
    }

    // Actor update
    const dqda = gradQWrtA(state, action, currentW);
    const dmuDtheta = gradMuWithTheta(state, currentTheta);
    const actorGradient = dmuDtheta.map((g) => actorAlpha * dqda * g);
    for (let i = 0; i < currentTheta.length; i++) {
      currentTheta[i] += actorGradient[i];
    }

    const stepRecord: DeterministicACStep = {
      step: t,
      state,
      action,
      reward: result.reward,
      nextState: result.nextState,
      done: result.done,
      nextAction,
      target,
      prediction,
      tdError,
      dqda,
      dmuDtheta,
      actorGradient,
      wBefore,
      wAfter: [...currentW],
      thetaBefore,
      thetaAfter: [...currentTheta],
    };

    if (!diverged) {
      const bad =
        !finite(tdError) ||
        !finite(dqda) ||
        !allFinite(actorGradient) ||
        !allFinite(currentW) ||
        !allFinite(currentTheta);
      if (bad) {
        diverged = true;
        divergenceStep = t;
      }
    }
    steps.push(stepRecord);

    state = result.nextState;
    if (result.done || env.terminal(state)) break;
  }

  return { steps, finalW: currentW, finalTheta: currentTheta, diverged, divergenceStep };
}
