/**
 * Policy gradient methods for Chapter 9.
 *
 * Implements the feature-based softmax policy
 *   h_theta(s,a) = theta_a^T phi(s)
 *   pi_theta(a|s) = softmax( h_theta(s, *) )
 * and REINFORCE / REINFORCE-with-baseline for both bandits and GridWorld MDPs.
 */

import { mulberry32 } from './stochasticApproximation';
import {
  type GridWorldConfig,
  type Policy,
  type Action,
  type StateValues,
  step,
  rewardForAction,
  sampleActionWithRng,
  solveStateValues,
  isTerminal,
} from './gridworld';
import { stationaryDistribution } from './fa';

export { stationaryDistribution };

// ---------------------------------------------------------------------------
// Feature constructors for the policy network h_theta(s,a) = theta_a^T phi(s)
// ---------------------------------------------------------------------------

export type PGFeatureMode = 'onehot' | 'coordinate' | 'distance';

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, x, i) => sum + x * (b[i] ?? 0), 0);
}

export function stateFeatures(
  state: number,
  mode: PGFeatureMode,
  config: GridWorldConfig
): number[] {
  if (mode === 'onehot') {
    const vec = new Array(config.rows * config.cols).fill(0);
    vec[state] = 1;
    return vec;
  }

  const row = Math.floor(state / config.cols);
  const col = state % config.cols;
  const rNorm = config.rows > 1 ? (row / (config.rows - 1)) * 2 - 1 : 0;
  const cNorm = config.cols > 1 ? (col / (config.cols - 1)) * 2 - 1 : 0;

  if (mode === 'coordinate') {
    return [1, rNorm, cNorm];
  }

  // distance
  const tRow = Math.floor(config.targetState / config.cols);
  const tCol = config.targetState % config.cols;
  const distanceToTarget = Math.sqrt((row - tRow) ** 2 + (col - tCol) ** 2);
  const maxDist = Math.sqrt((config.rows - 1) ** 2 + (config.cols - 1) ** 2);
  const isForbidden = config.forbiddenStates.includes(state) ? 1 : 0;
  return [1, rNorm, cNorm, distanceToTarget / Math.max(1, maxDist), isForbidden];
}

export function featureDim(mode: PGFeatureMode, config: GridWorldConfig): number {
  return stateFeatures(0, mode, config).length;
}

export function zeroTheta(numActions: number, featureDim: number): number[][] {
  return Array.from({ length: numActions }, () => new Array(featureDim).fill(0));
}

// ---------------------------------------------------------------------------
// Softmax policy and score function
// ---------------------------------------------------------------------------

export function softmaxPolicy(preferences: number[]): number[] {
  const maxPref = Math.max(...preferences);
  const exps = preferences.map((p) => Math.exp(p - maxPref));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export function policyPreferences(
  theta: number[][],
  state: number,
  mode: PGFeatureMode,
  config: GridWorldConfig
): number[] {
  const phi = stateFeatures(state, mode, config);
  return theta.map((w) => dot(w, phi));
}

export function policyTable(
  theta: number[][],
  mode: PGFeatureMode,
  config: GridWorldConfig
): Policy {
  const numStates = config.rows * config.cols;
  return Array.from({ length: numStates }, (_, s) =>
    softmaxPolicy(policyPreferences(theta, s, mode, config))
  );
}

/**
 * Tabular softmax score: ∂ log π(a|s) / ∂ h(s,a') = 1{a'=a} - π(a'|s).
 */
export function softmaxScoreGradient(policy: number[], action: number): number[] {
  return policy.map((_, j) => (j === action ? 1 - policy[j] : -policy[j]));
}

/**
 * Feature-based score: ∂ log π(a|s) / ∂ theta_a' = phi(s) * (1{a'=a} - π(a'|s)).
 * Returns a matrix [action][feature].
 */
export function softmaxScoreGradientFeature(
  policy: number[],
  action: number,
  phi: number[]
): number[][] {
  return policy.map((p, a) => phi.map((f) => f * (a === action ? 1 - p : -p)));
}

/**
 * Verify that Σ_a π(a|s) ∇log π(a|s) = 0 (tabular case).
 */
export function expectedScoreZero(policy: number[]): number[] {
  return policy.map((_, j) =>
    policy.reduce((sum, p, a) => sum + p * softmaxScoreGradient(policy, a)[j], 0)
  );
}

/**
 * Verify that Σ_a π(a|s) ∇log π(a|s) = 0 (feature-based case).
 * Returns a vector of length featureDim.
 */
export function expectedScoreZeroFeature(policy: number[], phi: number[]): number[] {
  const dim = phi.length;
  const result = new Array(dim).fill(0);
  for (let a = 0; a < policy.length; a++) {
    const score = softmaxScoreGradientFeature(policy, a, phi);
    for (let a2 = 0; a2 < policy.length; a2++) {
      for (let i = 0; i < dim; i++) {
        result[i] += policy[a] * score[a2][i];
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Discounted returns
// ---------------------------------------------------------------------------

export function computeDiscountedReturns(rewards: number[], gamma: number): number[] {
  const T = rewards.length;
  const returns = new Array(T).fill(0);
  let g = 0;
  for (let t = T - 1; t >= 0; t--) {
    g = rewards[t] + gamma * g;
    returns[t] = g;
  }
  return returns;
}

// ---------------------------------------------------------------------------
// Bandit helpers (phi(s) = [1], so h(a) = theta_a)
// ---------------------------------------------------------------------------

export interface BanditRecord {
  episode: number;
  action: number;
  reward: number;
  thetaBefore: number[];
  thetaAfter: number[];
  policyBefore: number[];
  policyAfter: number[];
  behaviorProb: number;
  updateProb: number;
  scoreGradient: number[];
  weightedGradient: number[];
  parameterDelta: number[];
  baselineBefore?: number;
  baselineAfter?: number;
  advantage?: number;
}

export interface BanditResult {
  thetaHistory: number[][];
  policyHistory: number[][];
  rewardHistory: number[];
  baselineHistory: number[];
  records: BanditRecord[];
}

function columnToFlat(theta: number[][]): number[] {
  return theta.map((w) => w[0]);
}

function flatToColumn(thetaFlat: number[]): number[][] {
  return thetaFlat.map((v) => [v]);
}

export function reinforceBandit(
  actionRewards: number[],
  initialTheta: number[],
  alpha: number,
  episodes: number,
  seed: number,
  rewardNoiseStd = 0.5
): BanditResult {
  const rng = mulberry32(seed);
  const theta = flatToColumn([...initialTheta]);

  const thetaHistory: number[][] = [columnToFlat(theta)];
  const policyHistory: number[][] = [softmaxPolicy(columnToFlat(theta))];
  const rewardHistory: number[] = [];
  const baselineHistory: number[] = [0];
  const records: BanditRecord[] = [];

  for (let ep = 0; ep < episodes; ep++) {
    const policyFlat = softmaxPolicy(columnToFlat(theta));
    const action = sampleActionWithRng(policyFlat, rng);
    const r = actionRewards[action] + (rng() * 2 - 1) * rewardNoiseStd;
    rewardHistory.push(r);

    const thetaBefore = columnToFlat(theta);
    const policyBefore = [...policyFlat];
    const score = softmaxScoreGradient(policyFlat, action);
    const weighted = score.map((g) => r * g);
    const delta = weighted.map((g) => alpha * g);

    for (let a = 0; a < theta.length; a++) {
      theta[a][0] += delta[a];
    }

    const policyAfter = softmaxPolicy(columnToFlat(theta));
    records.push({
      episode: ep,
      action,
      reward: r,
      thetaBefore,
      thetaAfter: columnToFlat(theta),
      policyBefore,
      policyAfter,
      behaviorProb: policyBefore[action],
      updateProb: policyBefore[action],
      scoreGradient: score,
      weightedGradient: weighted,
      parameterDelta: delta,
    });

    thetaHistory.push(columnToFlat(theta));
    policyHistory.push(policyAfter);
    baselineHistory.push(0);
  }

  return { thetaHistory, policyHistory, rewardHistory, baselineHistory, records };
}

export function reinforceWithBaseline(
  actionRewards: number[],
  initialTheta: number[],
  alpha: number,
  beta: number,
  episodes: number,
  seed: number,
  rewardNoiseStd = 0.5
): BanditResult {
  const rng = mulberry32(seed);
  const theta = flatToColumn([...initialTheta]);
  let baseline = 0;

  const thetaHistory: number[][] = [columnToFlat(theta)];
  const policyHistory: number[][] = [softmaxPolicy(columnToFlat(theta))];
  const rewardHistory: number[] = [];
  const baselineHistory: number[] = [baseline];
  const records: BanditRecord[] = [];

  for (let ep = 0; ep < episodes; ep++) {
    const policyFlat = softmaxPolicy(columnToFlat(theta));
    const action = sampleActionWithRng(policyFlat, rng);
    const r = actionRewards[action] + (rng() * 2 - 1) * rewardNoiseStd;
    rewardHistory.push(r);

    const baselineBefore = baseline;
    const advantage = r - baselineBefore;

    const thetaBefore = columnToFlat(theta);
    const policyBefore = [...policyFlat];
    const score = softmaxScoreGradient(policyFlat, action);
    const weighted = score.map((g) => advantage * g);
    const delta = weighted.map((g) => alpha * g);

    for (let a = 0; a < theta.length; a++) {
      theta[a][0] += delta[a];
    }

    // Baseline is updated AFTER the actor update, using the baseline that was
    // active for the current advantage.
    baseline += beta * (r - baselineBefore);

    const policyAfter = softmaxPolicy(columnToFlat(theta));
    records.push({
      episode: ep,
      action,
      reward: r,
      thetaBefore,
      thetaAfter: columnToFlat(theta),
      policyBefore,
      policyAfter,
      behaviorProb: policyBefore[action],
      updateProb: policyBefore[action],
      scoreGradient: score,
      weightedGradient: weighted,
      parameterDelta: delta,
      baselineBefore,
      baselineAfter: baseline,
      advantage,
    });

    thetaHistory.push(columnToFlat(theta));
    policyHistory.push(policyAfter);
    baselineHistory.push(baseline);
  }

  return { thetaHistory, policyHistory, rewardHistory, baselineHistory, records };
}

// ---------------------------------------------------------------------------
// MDP trajectory and REINFORCE
// ---------------------------------------------------------------------------

export interface TrajectoryStep {
  state: number;
  action: Action;
  reward: number;
  nextState: number;
  done: boolean;
}

export interface MDPStepDetail {
  state: number;
  action: Action;
  reward: number;
  behaviorProb: number;
  updateProb: number;
  returnGt: number;
  scoreGradient: number[][];
  weightedGradient: number[][];
  parameterDelta: number[][];
  baselineBefore?: number;
  baselineAfter?: number;
  advantage?: number;
}

export interface MDPEpisode {
  trajectory: TrajectoryStep[];
  cumulativeReward: number;
  discountedReturnG0: number;
  episodeLength: number;
  success: boolean;
  truncated: boolean;
  stepDetails: MDPStepDetail[];
}

export interface PGUpdateRecord {
  episode: number;
  time: number;
  state: number;
  action: Action;
  reward: number;
  nextState: number;
  done: boolean;
  returnGt: number;
  thetaBefore: number[][];
  thetaAfter: number[][];
  behaviorPolicy: Policy;
  updatePolicy: Policy;
  behaviorProb: number;
  updateProb: number;
  scoreGradient: number[][];
  weightedGradient: number[][];
  parameterDelta: number[][];
  baselineBefore?: number;
  baselineAfter?: number;
  advantage?: number;
}

export interface ReinforceMDPOptions {
  alpha: number;
  beta?: number;
  episodes: number;
  maxSteps: number;
  seed?: number;
  useBaseline?: boolean;
  featureMode?: PGFeatureMode;
}

export interface ReinforceMDPResult {
  thetaHistory: number[][][];
  policyHistory: Policy[];
  episodes: MDPEpisode[];
  updateRecords: PGUpdateRecord[];
}

export function reinforceMDP(
  config: GridWorldConfig,
  options: ReinforceMDPOptions
): ReinforceMDPResult {
  const {
    alpha,
    beta = 0.1,
    episodes,
    maxSteps,
    seed = 1,
    useBaseline = false,
    featureMode = 'onehot',
  } = options;
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;
  const fdim = featureDim(featureMode, config);
  let theta: number[][] = zeroTheta(numActions, fdim);
  const baseline: number[] = new Array(numStates).fill(0);

  const thetaHistory: number[][][] = [theta.map((row) => [...row])];
  const policyHistory: Policy[] = [policyTable(theta, featureMode, config)];
  const episodeDetails: MDPEpisode[] = [];
  const updateRecords: PGUpdateRecord[] = [];

  for (let ep = 0; ep < episodes; ep++) {
    const trajectory: TrajectoryStep[] = [];
    const behaviorPolicies: Policy[] = [];
    let state = config.startState;
    let cumulativeReward = 0;
    let terminal = false;

    for (let stepIdx = 0; stepIdx < maxSteps; stepIdx++) {
      if (isTerminal(state, config)) {
        terminal = true;
        break;
      }
      const behaviorPolicy = policyTable(theta, featureMode, config);
      const statePolicy = behaviorPolicy[state];
      behaviorPolicies.push(behaviorPolicy);
      const action = sampleActionWithRng(statePolicy, rng);
      const result = step(state, action, config);
      trajectory.push({
        state,
        action,
        reward: result.reward,
        nextState: result.nextState,
        done: result.done,
      });
      cumulativeReward += result.reward;
      state = result.nextState;
      if (result.done) {
        terminal = true;
        break;
      }
    }

    const rewards = trajectory.map((tr) => tr.reward);
    const returns = computeDiscountedReturns(rewards, config.gamma);
    const discountedReturnG0 = returns.length > 0 ? returns[0] : 0;
    const success = terminal && trajectory.length > 0 && trajectory[trajectory.length - 1].done;
    const truncated = !terminal && trajectory.length === maxSteps;

    const stepDetails: MDPStepDetail[] = [];
    for (let t = 0; t < trajectory.length; t++) {
      const { state: s, action: a, reward: r, nextState: ns, done } = trajectory[t];
      const behaviorPolicy = behaviorPolicies[t];
      const updatePolicy = policyTable(theta, featureMode, config);
      const behaviorProb = behaviorPolicy[s][a];
      const updateProb = updatePolicy[s][a];
      const phi = stateFeatures(s, featureMode, config);
      const score = softmaxScoreGradientFeature(updatePolicy[s], a, phi);

      const baselineBefore = baseline[s];
      const advantage = useBaseline ? returns[t] - baselineBefore : undefined;
      const weight = useBaseline ? advantage! : returns[t];
      const weighted = score.map((row) => row.map((g) => weight * g));
      const delta = weighted.map((row) => row.map((g) => alpha * g));

      for (let actionIdx = 0; actionIdx < numActions; actionIdx++) {
        for (let i = 0; i < fdim; i++) {
          theta[actionIdx][i] += delta[actionIdx][i];
        }
      }

      if (useBaseline) {
        baseline[s] += beta * (returns[t] - baselineBefore);
      }

      stepDetails.push({
        state: s,
        action: a,
        reward: r,
        behaviorProb,
        updateProb,
        returnGt: returns[t],
        scoreGradient: score,
        weightedGradient: weighted,
        parameterDelta: delta,
        baselineBefore: useBaseline ? baselineBefore : undefined,
        baselineAfter: useBaseline ? baseline[s] : undefined,
        advantage,
      });

      updateRecords.push({
        episode: ep,
        time: t,
        state: s,
        action: a,
        reward: r,
        nextState: ns,
        done,
        returnGt: returns[t],
        thetaBefore: theta.map((row, actionIdx) =>
          row.map((v, i) => (delta[actionIdx] ? v - delta[actionIdx][i] : v))
        ),
        thetaAfter: theta.map((row) => [...row]),
        behaviorPolicy,
        updatePolicy,
        behaviorProb,
        updateProb,
        scoreGradient: score,
        weightedGradient: weighted,
        parameterDelta: delta,
        baselineBefore: useBaseline ? baselineBefore : undefined,
        baselineAfter: useBaseline ? baseline[s] : undefined,
        advantage,
      });
    }

    episodeDetails.push({
      trajectory,
      cumulativeReward,
      discountedReturnG0,
      episodeLength: trajectory.length,
      success,
      truncated,
      stepDetails,
    });
    thetaHistory.push(theta.map((row) => [...row]));
    policyHistory.push(policyTable(theta, featureMode, config));
  }

  return { thetaHistory, policyHistory, episodes: episodeDetails, updateRecords };
}

export function reinforceMDPWithBaseline(
  config: GridWorldConfig,
  options: Omit<ReinforceMDPOptions, 'useBaseline'>
): ReinforceMDPResult {
  return reinforceMDP(config, { ...options, useBaseline: true });
}

// ---------------------------------------------------------------------------
// Objective metrics (Section 9.2)
// ---------------------------------------------------------------------------

export function expectedImmediateReward(
  state: number,
  policyState: number[],
  config: GridWorldConfig
): number {
  return policyState.reduce((sum, p, a) => sum + p * rewardForAction(state, a as Action, config), 0);
}

export interface PolicyMetrics {
  vPi: StateValues;
  rPi: number[];
  dPi: number[];
  Jv0: number;
  Jv: number;
  Jr: number;
  policy: Policy;
}

export function computePolicyMetrics(
  theta: number[][],
  config: GridWorldConfig,
  d0: number[],
  featureMode: PGFeatureMode
): PolicyMetrics {
  const policy = policyTable(theta, featureMode, config);
  const vPi = solveStateValues(policy, config);
  const rPi = policy.map((dist, s) => expectedImmediateReward(s, dist, config));
  const dPi = stationaryDistribution(policy, config, 2000);

  const Jv0 = d0.reduce((sum, d, s) => sum + d * vPi[s], 0);
  const Jv = dPi.reduce((sum, d, s) => sum + d * vPi[s], 0);
  const Jr = dPi.reduce((sum, d, s) => sum + d * rPi[s], 0);

  return { vPi, rPi, dPi, Jv0, Jv, Jr, policy };
}

// ---------------------------------------------------------------------------
// Variance reduction experiment (common random numbers)
// ---------------------------------------------------------------------------

export interface GradientEstimateBatch {
  seed: number;
  action: number;
  reward: number;
  gNoBaseline: number[];
  gBaseline: number[];
}

export interface VarianceComparisonResult {
  estimates: GradientEstimateBatch[];
  meanNoBaseline: number[];
  meanBaseline: number[];
  varNoBaseline: number[];
  varBaseline: number[];
  stdNoBaseline: number[];
  stdBaseline: number[];
  meanNormNoBaseline: number;
  meanNormBaseline: number;
  varNormNoBaseline: number;
  varNormBaseline: number;
}

function vectorNorm(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

function meanAndVar(values: number[]): { mean: number; variance: number } {
  const mean = values.reduce((s, x) => s + x, 0) / values.length;
  const variance = values.reduce((s, x) => s + (x - mean) ** 2, 0) / values.length;
  return { mean, variance };
}

export function compareBaselineVariance(
  theta: number[],
  actionRewards: number[],
  baseline: number,
  numSeeds: number,
  rewardNoiseStd = 0.5
): VarianceComparisonResult {
  const estimates: GradientEstimateBatch[] = [];
  const dim = theta.length;

  for (let seed = 1; seed <= numSeeds; seed++) {
    const rng = mulberry32(seed);
    const policy = softmaxPolicy(theta);
    const action = sampleActionWithRng(policy, rng);
    const r = actionRewards[action] + (rng() * 2 - 1) * rewardNoiseStd;
    const score = softmaxScoreGradient(policy, action);
    const gNoBaseline = score.map((g) => r * g);
    const gBaseline = score.map((g) => (r - baseline) * g);
    estimates.push({ seed, action, reward: r, gNoBaseline, gBaseline });
  }

  const meanNoBaseline = new Array(dim).fill(0);
  const meanBaseline = new Array(dim).fill(0);
  estimates.forEach((e) => {
    e.gNoBaseline.forEach((v, i) => (meanNoBaseline[i] += v));
    e.gBaseline.forEach((v, i) => (meanBaseline[i] += v));
  });
  meanNoBaseline.forEach((_, i) => (meanNoBaseline[i] /= estimates.length));
  meanBaseline.forEach((_, i) => (meanBaseline[i] /= estimates.length));

  const varNoBaseline = new Array(dim).fill(0);
  const varBaseline = new Array(dim).fill(0);
  estimates.forEach((e) => {
    e.gNoBaseline.forEach((v, i) => {
      const d = v - meanNoBaseline[i];
      varNoBaseline[i] += d * d;
    });
    e.gBaseline.forEach((v, i) => {
      const d = v - meanBaseline[i];
      varBaseline[i] += d * d;
    });
  });
  varNoBaseline.forEach((_, i) => (varNoBaseline[i] /= estimates.length));
  varBaseline.forEach((_, i) => (varBaseline[i] /= estimates.length));

  const normsNoBaseline = estimates.map((e) => vectorNorm(e.gNoBaseline));
  const normsBaseline = estimates.map((e) => vectorNorm(e.gBaseline));
  const normStatsNoBaseline = meanAndVar(normsNoBaseline);
  const normStatsBaseline = meanAndVar(normsBaseline);

  return {
    estimates,
    meanNoBaseline,
    meanBaseline,
    varNoBaseline,
    varBaseline,
    stdNoBaseline: varNoBaseline.map(Math.sqrt),
    stdBaseline: varBaseline.map(Math.sqrt),
    meanNormNoBaseline: normStatsNoBaseline.mean,
    meanNormBaseline: normStatsBaseline.mean,
    varNormNoBaseline: normStatsNoBaseline.variance,
    varNormBaseline: normStatsBaseline.variance,
  };
}
