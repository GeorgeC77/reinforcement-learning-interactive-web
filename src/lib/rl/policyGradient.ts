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
  computeBellmanComponents,
  solveLinearSystem as solveLinearSystemGW,
  isTerminal,
} from './gridworld';

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
 * Returns a matrix [parameterAction][feature].
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
 * Returns a matrix [parameterAction][feature].
 */
export function expectedScoreZeroFeature(
  policy: number[],
  phi: number[]
): number[][] {
  const numActions = policy.length;
  const result = Array.from(
    { length: numActions },
    () => new Array(phi.length).fill(0)
  );

  for (let sampledAction = 0; sampledAction < numActions; sampledAction++) {
    const score = softmaxScoreGradientFeature(policy, sampledAction, phi);
    for (let parameterAction = 0; parameterAction < numActions; parameterAction++) {
      for (let j = 0; j < phi.length; j++) {
        result[parameterAction][j] += policy[sampledAction] * score[parameterAction][j];
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
  updateProbBefore: number;
  updateProbAfter: number;
  deltaProbability: number;
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
  policyBeforeUpdate: Policy;
  policyAfterUpdate: Policy;
  behaviorProb: number;
  updateProbBefore: number;
  updateProbAfter: number;
  deltaProbability: number;
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
      const behaviorProb = behaviorPolicy[s][a];

      const thetaBefore = theta.map((row) => [...row]);
      const policyBeforeUpdate = policyTable(thetaBefore, featureMode, config);
      const updateProbBefore = policyBeforeUpdate[s][a];

      const phi = stateFeatures(s, featureMode, config);
      const score = softmaxScoreGradientFeature(policyBeforeUpdate[s], a, phi);

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

      const thetaAfter = theta.map((row) => [...row]);
      const policyAfterUpdate = policyTable(thetaAfter, featureMode, config);
      const updateProbAfter = policyAfterUpdate[s][a];
      const deltaProbability = updateProbAfter - updateProbBefore;

      stepDetails.push({
        state: s,
        action: a,
        reward: r,
        behaviorProb,
        updateProbBefore,
        updateProbAfter,
        deltaProbability,
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
        thetaBefore,
        thetaAfter,
        behaviorPolicy,
        policyBeforeUpdate,
        policyAfterUpdate,
        behaviorProb,
        updateProbBefore,
        updateProbAfter,
        deltaProbability,
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

export interface StationaryDistributionResult {
  d: number[];
  residual: number;
  converged: boolean;
  iterations: number;
}

export function computeStationaryDistribution(
  policy: Policy,
  config: GridWorldConfig,
  maxIter = 5000,
  tol = 1e-10
): StationaryDistributionResult {
  const numStates = config.rows * config.cols;
  let d = new Array(numStates).fill(1 / numStates);
  let residual = Infinity;
  let converged = false;
  let iterations = 0;

  for (let it = 0; it < maxIter; it++) {
    iterations = it + 1;
    const next = new Array(numStates).fill(0);
    for (let s = 0; s < numStates; s++) {
      for (let a = 0; a < 5; a++) {
        const result = step(s, a as Action, config);
        next[result.nextState] += d[s] * policy[s][a];
      }
    }
    const sum = next.reduce((acc, v) => acc + v, 0);
    if (sum === 0) break;
    const norm = next.map((v) => v / sum);
    residual = Math.max(...norm.map((v, i) => Math.abs(v - d[i])));
    d = norm;
    if (residual < tol) {
      converged = true;
      break;
    }
  }

  return { d, residual, converged, iterations };
}

export interface PolicyMetrics {
  vPi: StateValues;
  rPi: number[];
  dPi: number[];
  d0: number[];
  Jv0: number;
  Jv: number;
  Jr: number;
  relationResidual: number;
  stationaryResidual: number;
  stationaryConverged: boolean;
  stationaryIterations: number;
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
  const { d, residual, converged, iterations } = computeStationaryDistribution(policy, config);

  const Jv0 = d0.reduce((sum, dVal, s) => sum + dVal * vPi[s], 0);
  const Jv = d.reduce((sum, dVal, s) => sum + dVal * vPi[s], 0);
  const Jr = d.reduce((sum, dVal, s) => sum + dVal * rPi[s], 0);
  const relationResidual = Math.abs(Jr - (1 - config.gamma) * Jv);

  return {
    vPi,
    rPi,
    dPi: d,
    d0,
    Jv0,
    Jv,
    Jr,
    relationResidual,
    stationaryResidual: residual,
    stationaryConverged: converged,
    stationaryIterations: iterations,
    policy,
  };
}

// ---------------------------------------------------------------------------
// Exact discounted policy-gradient checker (Theorem 9.2)
// ---------------------------------------------------------------------------

export function computeTransitionMatrix(
  policy: Policy,
  config: GridWorldConfig
): number[][] {
  const numStates = config.rows * config.cols;
  const P: number[][] = Array.from({ length: numStates }, () => new Array(numStates).fill(0));
  for (let s = 0; s < numStates; s++) {
    for (let a = 0; a < 5; a++) {
      const result = step(s, a as Action, config);
      P[result.nextState][s] += policy[s][a];
    }
  }
  return P;
}

/**
 * Discounted occupancy measure ρ_π (unnormalized).
 * Solves ρ = d0 + γ P^T ρ, where sum(ρ) = 1/(1-γ) when sum(d0)=1.
 */
export function computeDiscountOccupancy(
  policy: Policy,
  config: GridWorldConfig,
  d0: number[],
  gamma: number,
  maxIter = 5000,
  tol = 1e-12
): number[] {
  const P = computeTransitionMatrix(policy, config);
  const numStates = P.length;
  let rho = [...d0];
  for (let it = 0; it < maxIter; it++) {
    const next = new Array(numStates).fill(0);
    for (let s = 0; s < numStates; s++) {
      let acc = d0[s];
      for (let sp = 0; sp < numStates; sp++) {
        acc += gamma * P[s][sp] * rho[sp];
      }
      next[s] = acc;
    }
    const diff = Math.max(...next.map((v, i) => Math.abs(v - rho[i])));
    rho = next;
    if (diff < tol) break;
  }
  return rho;
}

/**
 * Exact gradient of J(θ) = Σ_s d0(s) v_π(s) for one-hot softmax policy.
 * Returns the gradient component for a single (parameterAction, parameterState).
 */
function solveStateValuesGamma(policy: Policy, config: GridWorldConfig, gamma: number): StateValues {
  const { rPi, pPi } = computeBellmanComponents(policy, config);
  const numStates = rPi.length;
  const A = Array.from({ length: numStates }, (_, i) =>
    Array.from({ length: numStates }, (_, j) => (i === j ? 1 : 0) - gamma * pPi[i][j])
  );
  return solveLinearSystemGW(A, rPi);
}

function computeQValuesGamma(values: StateValues, config: GridWorldConfig, gamma: number): number[][] {
  const numStates = config.rows * config.cols;
  const q: number[][] = Array.from({ length: numStates }, () => new Array(5).fill(0));
  for (let s = 0; s < numStates; s++) {
    if (isTerminal(s, config)) {
      q[s] = new Array(5).fill(0);
      continue;
    }
    for (let a = 0; a < 5; a++) {
      const result = step(s, a as Action, config);
      q[s][a] = result.done ? result.reward : result.reward + gamma * values[result.nextState];
    }
  }
  return q;
}

export function exactDiscountGradientComponent(
  theta: number[][],
  config: GridWorldConfig,
  d0: number[],
  gamma: number,
  parameterAction: number,
  parameterState: number
): number {
  const policy = policyTable(theta, 'onehot', config);
  const vPi = solveStateValuesGamma(policy, config, gamma);
  const qPi = computeQValuesGamma(vPi, config, gamma);
  const rho = computeDiscountOccupancy(policy, config, d0, gamma);
  const s = parameterState;
  const a = parameterAction;
  const piA = policy[s][a];
  const vS = vPi[s];
  const qA = qPi[s][a];
  // ∂J/∂θ_{a,s} = ρ(s) π(a|s) (q(s,a) - v(s))
  return rho[s] * piA * (qA - vS);
}

export function finiteDifferenceGradientComponent(
  theta: number[][],
  config: GridWorldConfig,
  d0: number[],
  gamma: number,
  parameterAction: number,
  parameterState: number,
  eps = 1e-4
): number {
  function objective(th: number[][]) {
    const pol = policyTable(th, 'onehot', config);
    const vals = solveStateValuesGamma(pol, config, gamma);
    return d0.reduce((sum, dVal, s) => sum + dVal * vals[s], 0);
  }

  const thetaPlus = theta.map((row, a) =>
    row.map((v, s) => (a === parameterAction && s === parameterState ? v + eps : v))
  );
  const thetaMinus = theta.map((row, a) =>
    row.map((v, s) => (a === parameterAction && s === parameterState ? v - eps : v))
  );
  return (objective(thetaPlus) - objective(thetaMinus)) / (2 * eps);
}

export function stationaryApproxGradientComponent(
  theta: number[][],
  config: GridWorldConfig,
  gamma: number,
  parameterAction: number,
  parameterState: number
): number {
  const policy = policyTable(theta, 'onehot', config);
  const vPi = solveStateValuesGamma(policy, config, gamma);
  const qPi = computeQValuesGamma(vPi, config, gamma);
  const { d } = computeStationaryDistribution(policy, config);
  const s = parameterState;
  const a = parameterAction;
  // ∇v̄_π ≈ (1/(1-γ)) E_{S~dπ}[Σ_a π(a|s) ∇logπ(a|s) q(s,a)]
  return (1 / (1 - gamma)) * d[s] * policy[s][a] * (qPi[s][a] - vPi[s]);
}

export interface DiscountGradientCheckResult {
  finiteDifference: number;
  exact: number;
  stationaryApprox: number;
  exactError: number;
  stationaryError: number;
  rhoSum: number;
  expectedRhoSum: number;
}

export function checkDiscountGradientComponent(
  theta: number[][],
  config: GridWorldConfig,
  d0: number[],
  gamma: number,
  parameterAction: number,
  parameterState: number,
  eps = 1e-4
): DiscountGradientCheckResult {
  const finiteDifference = finiteDifferenceGradientComponent(
    theta,
    config,
    d0,
    gamma,
    parameterAction,
    parameterState,
    eps
  );
  const exact = exactDiscountGradientComponent(
    theta,
    config,
    d0,
    gamma,
    parameterAction,
    parameterState
  );
  const stationaryApprox = stationaryApproxGradientComponent(
    theta,
    config,
    gamma,
    parameterAction,
    parameterState
  );
  const policy = policyTable(theta, 'onehot', config);
  const rho = computeDiscountOccupancy(policy, config, d0, gamma);
  return {
    finiteDifference,
    exact,
    stationaryApprox,
    exactError: Math.abs(finiteDifference - exact),
    stationaryError: Math.abs(finiteDifference - stationaryApprox),
    rhoSum: rho.reduce((s, v) => s + v, 0),
    expectedRhoSum: 1 / (1 - gamma),
  };
}

export function checkDiscountGradientOverGamma(
  theta: number[][],
  config: GridWorldConfig,
  d0: number[],
  parameterAction: number,
  parameterState: number,
  gammas: number[]
): {
  gamma: number;
  exactError: number;
  stationaryError: number;
}[] {
  return gammas.map((gamma) => {
    const check = checkDiscountGradientComponent(
      theta,
      config,
      d0,
      gamma,
      parameterAction,
      parameterState
    );
    return { gamma, exactError: check.exactError, stationaryError: check.stationaryError };
  });
}

// ---------------------------------------------------------------------------
// Average-reward / differential value metrics
// ---------------------------------------------------------------------------

export interface AverageRewardMetrics {
  rBar: number;
  dPi: number[];
  differentialV: number[];
  differentialQ: number[][];
  poissonResidual: number;
  ordinaryCumulative: number[];
  differentialCumulative: number[];
}

export function computeAverageRewardMetrics(
  policy: Policy,
  config: GridWorldConfig,
  horizon = 50,
  seed = 1
): AverageRewardMetrics {
  const numStates = config.rows * config.cols;
  const rPi = policy.map((dist, s) => expectedImmediateReward(s, dist, config));
  const { d } = computeStationaryDistribution(policy, config);
  const rBar = d.reduce((sum, ds, s) => sum + ds * rPi[s], 0);

  // Poisson equation: (I - P^T) v = r - rBar 1, with d^T v = 0.
  const P = computeTransitionMatrix(policy, config);
  const A: number[][] = Array.from({ length: numStates }, () => new Array(numStates).fill(0));
  const B: number[] = new Array(numStates).fill(0);
  for (let s = 0; s < numStates; s++) {
    A[s][s] = 1;
    for (let sp = 0; sp < numStates; sp++) {
      A[s][sp] -= P[sp][s];
    }
    B[s] = rPi[s] - rBar;
  }
  // Replace the last row with the normalization condition d^T v = 0.
  for (let s = 0; s < numStates; s++) {
    A[numStates - 1][s] = d[s];
  }
  B[numStates - 1] = 0;

  const solution = solveLinearSystemGW(A, B);
  const differentialV = solution;

  const differentialQ: number[][] = Array.from({ length: numStates }, () => new Array(5).fill(0));
  for (let s = 0; s < numStates; s++) {
    for (let a = 0; a < 5; a++) {
      const result = step(s, a as Action, config);
      const nextV = result.done ? 0 : differentialV[result.nextState];
      differentialQ[s][a] = result.reward - rBar + nextV;
    }
  }

  const Pv = new Array(numStates).fill(0);
  for (let s = 0; s < numStates; s++) {
    for (let sp = 0; sp < numStates; sp++) {
      Pv[s] += P[sp][s] * differentialV[sp];
    }
  }
  const poissonResidual = Math.max(
    ...differentialV.map((v, s) => Math.abs(v - Pv[s] - (rPi[s] - rBar)))
  );

  // Simulate one trajectory from start for ordinary vs differential cumulative rewards.
  const rng = mulberry32(seed);
  let state = config.startState;
  const ordinaryCumulative: number[] = [];
  const differentialCumulative: number[] = [];
  let ordinarySum = 0;
  let differentialSum = 0;
  for (let t = 0; t < horizon; t++) {
    const action = sampleActionWithRng(policy[state], rng);
    const result = step(state, action, config);
    ordinarySum += result.reward;
    differentialSum += result.reward - rBar;
    ordinaryCumulative.push(ordinarySum);
    differentialCumulative.push(differentialSum);
    state = result.nextState;
    if (result.done) break;
  }

  return {
    rBar,
    dPi: d,
    differentialV,
    differentialQ,
    poissonResidual,
    ordinaryCumulative,
    differentialCumulative,
  };
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

export interface BaselineMethodStats {
  name: string;
  baseline: number;
  meanGradient: number[];
  traceCov: number;
  meanNorm: number;
  varNorm: number;
}

export interface BaselineComparisonResult {
  methods: BaselineMethodStats[];
}

/**
 * Compare no baseline, a user baseline, the mean-return baseline, and the
 * score-weighted optimal scalar baseline
 *   b* = E[G ||score||^2] / E[||score||^2]
 *
 * Trace covariance = E ||g - E[g]||^2 = sum_i Var(g_i).
 */
export function compareMultipleBaselines(
  theta: number[],
  actionRewards: number[],
  userBaseline: number,
  numSeeds: number,
  rewardNoiseStd = 0.5
): BaselineComparisonResult {
  const dim = theta.length;
  const samples: { reward: number; score: number[]; scoreNormSq: number }[] = [];

  for (let seed = 1; seed <= numSeeds; seed++) {
    const rng = mulberry32(seed);
    const policy = softmaxPolicy(theta);
    const action = sampleActionWithRng(policy, rng);
    const r = actionRewards[action] + (rng() * 2 - 1) * rewardNoiseStd;
    const score = softmaxScoreGradient(policy, action);
    const scoreNormSq = score.reduce((s, x) => s + x * x, 0);
    samples.push({ reward: r, score, scoreNormSq });
  }

  const meanReward = samples.reduce((s, x) => s + x.reward, 0) / samples.length;
  const bStar =
    samples.reduce((s, x) => s + x.reward * x.scoreNormSq, 0) /
    samples.reduce((s, x) => s + x.scoreNormSq, 0);

  const configs = [
    { name: '无基线', baseline: 0 },
    { name: '用户基线', baseline: userBaseline },
    { name: '平均回报基线', baseline: meanReward },
    { name: '最优标量基线 b*', baseline: bStar },
  ];

  const methods = configs.map(({ name, baseline }) => {
    const grads = samples.map(({ reward, score }) => score.map((s) => (reward - baseline) * s));
    const meanGradient = new Array(dim).fill(0);
    grads.forEach((g) => g.forEach((v, i) => (meanGradient[i] += v)));
    meanGradient.forEach((_, i) => (meanGradient[i] /= grads.length));

    const traceCov = meanGradient.reduce((sum, _m, i) => {
      const mean = meanGradient[i];
      const variance = grads.reduce((s, g) => s + (g[i] - mean) ** 2, 0) / grads.length;
      return sum + variance;
    }, 0);

    const norms = grads.map((g) => vectorNorm(g));
    const normStats = meanAndVar(norms);

    return {
      name,
      baseline,
      meanGradient,
      traceCov,
      meanNorm: normStats.mean,
      varNorm: normStats.variance,
    };
  });

  return { methods };
}
