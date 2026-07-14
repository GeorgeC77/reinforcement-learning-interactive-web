/**
 * Unified GridWorld environment and RL utilities.
 *
 * Core gridworld environment and dynamic-programming utilities.
 *
 * Default 3x3 layout (row-major indices):
 *   s1(0) s2(1) s3(2)
 *   s4(3) s5(4) s6(5)
 *   s7(6) s8(7) s9(8)
 *
 * Default environment:
 *   - targetState:    s9  (index 8)
 *   - forbiddenStates: s6, s7 (indices [5, 6])
 *   - boundaryReward: -1
 *   - forbiddenReward: -1
 *   - targetReward:   +1
 *   - stepReward:     0
 *   - gamma:          0.9
 *   - taskType:       'continuing' for Chapters 1-4
 *
 * All algorithms in this file operate on the same `step()` function so that
 * boundary rewards, forbidden rewards, target rewards and episode termination
 * are handled consistently.
 */

import { mulberry32 } from './stochasticApproximation';

export type Action = 0 | 1 | 2 | 3 | 4;
export type StateIdx = number;
export type TaskType = 'continuing' | 'episodic';

export const ACTION_NAMES = ['上', '右', '下', '左', '停留'] as const;
export const ACTION_DELTAS: { dr: number; dc: number }[] = [
  { dr: -1, dc: 0 }, // up    (a1)
  { dr: 0, dc: 1 },  // right (a2)
  { dr: 1, dc: 0 },  // down  (a3)
  { dr: 0, dc: -1 }, // left  (a4)
  { dr: 0, dc: 0 },  // stay  (a5)
];

export type Policy = number[][]; // policy[s][a] = probability
export type StateValues = number[];

export interface StepResult {
  nextState: number;
  reward: number;
  done: boolean;
  info: {
    hitBoundary: boolean;
    enteredForbidden: boolean;
    reachedTarget: boolean;
    actionName: string;
    transitionText: string;
  };
}

export interface GridWorldConfig {
  rows: number;
  cols: number;
  startState: number;
  targetState: number;
  forbiddenStates: number[];
  targetReward: number;
  forbiddenReward: number;
  boundaryReward: number;
  stepReward: number;
  gamma: number;
  taskType: TaskType;
}

export const DEFAULT_CONFIG: GridWorldConfig = {
  rows: 3,
  cols: 3,
  startState: 0,
  targetState: 8,
  forbiddenStates: [5, 6], // s6 and s7
  targetReward: 1,
  forbiddenReward: -1,
  boundaryReward: -1,
  stepReward: 0,
  gamma: 0.9,
  taskType: 'continuing',
};

/** A milder boundary penalty useful for visualising policy trade-offs. */
export const MILD_BOUNDARY_CONFIG: GridWorldConfig = {
  ...DEFAULT_CONFIG,
  boundaryReward: -0.3,
};

/** GridWorld with target in the center-top (for varying optimal paths). */
export const CENTER_TARGET_CONFIG: GridWorldConfig = {
  ...DEFAULT_CONFIG,
  targetState: 4,
  forbiddenStates: [2],
};

/** High discount factor (far-sighted policies). */
export const FAR_SIGHTED_CONFIG: GridWorldConfig = {
  ...DEFAULT_CONFIG,
  gamma: 0.99,
};

/** Low discount factor (short-sighted policies). */
export const SHORT_SIGHTED_CONFIG: GridWorldConfig = {
  ...DEFAULT_CONFIG,
  gamma: 0.3,
};

/** Episodic task configuration often used in path-finding / TD chapters. */
export const EPISODIC_PATH_CONFIG: GridWorldConfig = {
  ...DEFAULT_CONFIG,
  targetReward: 0,
  forbiddenReward: -10,
  boundaryReward: -10,
  stepReward: -1,
  taskType: 'episodic',
};

export function stateToRowCol(state: number, cols: number): { row: number; col: number } {
  return { row: Math.floor(state / cols), col: state % cols };
}

export function rowColToState(row: number, col: number, cols: number): number {
  return row * cols + col;
}

/**
 * Intended next state under deterministic dynamics (boundary bounce-back).
 * Prefer `step()` for the complete transition because it returns the correct
 * boundary reward.
 */
export function nextState(state: number, action: Action, config: GridWorldConfig): number {
  const { row, col } = stateToRowCol(state, config.cols);
  const delta = ACTION_DELTAS[action];
  const newRow = row + delta.dr;
  const newCol = col + delta.dc;

  if (
    newRow < 0 ||
    newRow >= config.rows ||
    newCol < 0 ||
    newCol >= config.cols
  ) {
    return state; // boundary bounce back
  }
  return rowColToState(newRow, newCol, config.cols);
}

/**
 * Unified environment step. This is the single source of truth for transitions
 * and rewards used by all algorithms and UI components.
 */
export function step(
  state: number,
  action: Action,
  config: GridWorldConfig
): StepResult {
  const { row, col } = stateToRowCol(state, config.cols);
  const delta = ACTION_DELTAS[action];
  const newRow = row + delta.dr;
  const newCol = col + delta.dc;

  const hitBoundary =
    newRow < 0 ||
    newRow >= config.rows ||
    newCol < 0 ||
    newCol >= config.cols;

  const nextState = hitBoundary ? state : rowColToState(newRow, newCol, config.cols);
  const reachedTarget = !hitBoundary && nextState === config.targetState;
  const enteredForbidden = !hitBoundary && config.forbiddenStates.includes(nextState);

  let reward: number;
  if (hitBoundary) {
    reward = config.boundaryReward;
  } else if (nextState === config.targetState) {
    reward = config.targetReward;
  } else if (config.forbiddenStates.includes(nextState)) {
    reward = config.forbiddenReward;
  } else {
    reward = config.stepReward;
  }

  const done = config.taskType === 'episodic' && reachedTarget;
  const actionName = ACTION_NAMES[action];
  const transitionText = hitBoundary
    ? `s${state + 1} --${actionName}--> 撞边界，停留在 s${state + 1}，奖励 ${reward}`
    : `s${state + 1} --${actionName}--> s${nextState + 1}，奖励 ${reward}`;

  return {
    nextState,
    reward,
    done,
    info: {
      hitBoundary,
      enteredForbidden,
      reachedTarget,
      actionName,
      transitionText,
    },
  };
}

/**
 * State-based reward r(s) used for colouring cells. It does NOT include the
 * boundary reward because boundary is an action effect, not a state property.
 */
export function reward(state: number, config: GridWorldConfig): number {
  if (state === config.targetState) return config.targetReward;
  if (config.forbiddenStates.includes(state)) return config.forbiddenReward;
  return config.stepReward;
}

/**
 * Reward associated with taking an action at a state: r(s,a).
 * This correctly returns boundaryReward when the action attempts to leave the grid.
 */
export function rewardForAction(
  state: number,
  action: Action,
  config: GridWorldConfig
): number {
  return step(state, action, config).reward;
}

/**
 * Deterministic transition table for a state: list next states for each action.
 */
export function transitionTable(state: number, config: GridWorldConfig): number[] {
  return ACTION_DELTAS.map((_, a) => step(state, a as Action, config).nextState);
}

/**
 * Terminal test. In continuing tasks the target is a normal state.
 */
export function isTerminal(state: number, config: GridWorldConfig): boolean {
  return config.taskType === 'episodic' && state === config.targetState;
}

/**
 * Full stochastic outcome for a single step.
 * Keeps both the intended and the actually executed action so the UI can
 * illustrate when a slip occurs.  Reward and done come from the actual action.
 */
export interface StochasticOutcome {
  intendedAction: Action;
  actualAction: Action;
  nextState: number;
  reward: number;
  done: boolean;
  prob: number;
}

/**
 * Stochastic step distribution with full outcomes.
 * With probability (1 - slip) the intended action is executed;
 * with probability slip a uniformly random action (including the intended one) is executed.
 * Each outcome stores the actual action, next state, reward, and terminal flag.
 *
 * Outcomes are NOT merged by nextState, because different actual actions can yield
 * the same nextState but different rewards (e.g. boundary stay vs. real move).
 */
export function stochasticStepDistribution(
  state: number,
  intendedAction: Action,
  config: GridWorldConfig,
  slip: number
): StochasticOutcome[] {
  const numActions = ACTION_DELTAS.length;
  const intendedResult = step(state, intendedAction, config);
  const outcomes: StochasticOutcome[] = [
    {
      intendedAction,
      actualAction: intendedAction,
      nextState: intendedResult.nextState,
      reward: intendedResult.reward,
      done: intendedResult.done,
      prob: 1 - slip,
    },
  ];
  for (let a = 0; a < numActions; a++) {
    const actualAction = a as Action;
    const result = step(state, actualAction, config);
    outcomes.push({
      intendedAction,
      actualAction,
      nextState: result.nextState,
      reward: result.reward,
      done: result.done,
      prob: slip / numActions,
    });
  }
  return outcomes;
}

/**
 * Legacy next-state-only distribution (kept for visualisations that only need s').
 * Note: this merges outcomes with the same nextState, which can hide reward differences.
 */
export function stochasticTransition(
  state: number,
  action: Action,
  config: GridWorldConfig,
  slip: number
): { nextState: number; prob: number }[] {
  const counts = new Map<number, number>();
  for (const outcome of stochasticStepDistribution(state, action, config, slip)) {
    counts.set(outcome.nextState, (counts.get(outcome.nextState) ?? 0) + outcome.prob);
  }
  return Array.from(counts.entries())
    .map(([nextState, prob]) => ({ nextState, prob }))
    .sort((a, b) => a.nextState - b.nextState);
}

/**
 * Create a deterministic policy where each state takes the given action.
 */
export function deterministicPolicy(
  actions: Action[],
  numActions: number = 5
): Policy {
  return actions.map((a) => {
    const dist = new Array(numActions).fill(0);
    dist[a] = 1;
    return dist;
  });
}

/**
 * Uniform random policy.
 */
export function randomPolicy(numStates: number, numActions: number = 5): Policy {
  return Array.from({ length: numStates }, () => {
    const dist = new Array(numActions).fill(1 / numActions);
    return dist;
  });
}

/**
 * Normalize a policy so that probabilities in each state sum to 1.
 */
export function normalizePolicy(policy: Policy): Policy {
  return policy.map((dist) => {
    const sum = dist.reduce((a, b) => a + b, 0);
    if (sum === 0) {
      const uniform = 1 / dist.length;
      return dist.map(() => uniform);
    }
    return dist.map((p) => p / sum);
  });
}

export function sampleActionWithRng(
  probs: number[],
  rng: () => number
): Action {
  const r = rng();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r <= cum) return i as Action;
  }
  return (probs.length - 1) as Action;
}

export function sampleAction(probs: number[]): Action {
  return sampleActionWithRng(probs, Math.random);
}

/**
 * Compute the Bellman components for a given policy using the unified step().
 *   r_pi[s] = sum_a pi(a|s) * r(s,a)
 *   P_pi[s,s'] = sum_a pi(a|s) * p(s'|s,a)
 */
export function computeBellmanComponents(
  policy: Policy,
  config: GridWorldConfig
): { rPi: number[]; pPi: number[][] } {
  const numStates = config.rows * config.cols;
  const rPi: number[] = new Array(numStates).fill(0);
  const pPi: number[][] = Array.from({ length: numStates }, () => new Array(numStates).fill(0));

  for (let s = 0; s < numStates; s++) {
    if (isTerminal(s, config)) {
      rPi[s] = 0;
      continue;
    }

    for (let a = 0; a < 5; a++) {
      const prob = policy[s][a];
      if (prob === 0) continue;
      const result = step(s, a as Action, config);
      rPi[s] += prob * result.reward;
      // In episodic tasks, terminal transitions do not bootstrap to nextState.
      if (config.taskType === 'episodic' && result.done) continue;
      pPi[s][result.nextState] += prob;
    }
  }

  return { rPi, pPi };
}

/**
 * Decompose the Bellman backup for a single state:
 * v(s) = r_pi(s) + gamma * sum_{s'} P_pi(s,s') * v(s').
 */
export function computeBellmanBackup(
  state: number,
  values: StateValues,
  policy: Policy,
  config: GridWorldConfig
): {
  immediateReward: number;
  futureValue: number;
  backupValue: number;
  contributions: { nextState: number; prob: number; value: number; weightedValue: number }[];
} {
  const { rPi, pPi } = computeBellmanComponents(policy, config);
  const immediateReward = rPi[state];
  let futureValue = 0;
  const contributions: { nextState: number; prob: number; value: number; weightedValue: number }[] = [];
  for (let sNext = 0; sNext < values.length; sNext++) {
    const prob = pPi[state][sNext];
    if (prob === 0) continue;
    const weightedValue = prob * values[sNext];
    futureValue += weightedValue;
    contributions.push({ nextState: sNext, prob, value: values[sNext], weightedValue });
  }
  futureValue *= config.gamma;
  return { immediateReward, futureValue, backupValue: immediateReward + futureValue, contributions };
}

/**
 * Solve state values exactly: v = (I - gamma * P_pi)^{-1} * r_pi.
 */
export function solveStateValues(policy: Policy, config: GridWorldConfig): StateValues {
  const numStates = config.rows * config.cols;
  const { rPi, pPi } = computeBellmanComponents(policy, config);

  const A: number[][] = Array.from({ length: numStates }, (_, i) =>
    Array.from({ length: numStates }, (_, j) => (i === j ? 1 : 0) - config.gamma * pPi[i][j])
  );

  return solveLinearSystem(A, rPi);
}

export function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M: number[][] = A.map((row, i) => [...row, b[i]]);

  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
    }
    [M[i], M[maxRow]] = [M[maxRow], M[i]];

    const pivot = M[i][i];
    if (Math.abs(pivot) < 1e-12) {
      throw new Error('Singular matrix encountered while solving Bellman equation.');
    }

    for (let j = i; j <= n; j++) M[i][j] /= pivot;

    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const factor = M[k][i];
      for (let j = i; j <= n; j++) M[k][j] -= factor * M[i][j];
    }
  }

  return M.map((row) => row[n]);
}

/**
 * Iteratively solve state values using Bellman backup.
 */
export function iterateStateValues(
  policy: Policy,
  config: GridWorldConfig,
  iterations: number = 100,
  initialValues: StateValues | null = null
): StateValues[] {
  const numStates = config.rows * config.cols;
  const { rPi, pPi } = computeBellmanComponents(policy, config);
  let v = initialValues ? [...initialValues] : new Array(numStates).fill(0);
  const history: StateValues[] = [v];

  for (let k = 0; k < iterations; k++) {
    const vNext = new Array(numStates).fill(0);
    for (let s = 0; s < numStates; s++) {
      let sum = rPi[s];
      for (let sNext = 0; sNext < numStates; sNext++) {
        sum += config.gamma * pPi[s][sNext] * v[sNext];
      }
      vNext[s] = sum;
    }
    v = vNext;
    history.push([...v]);
  }

  return history;
}

/**
 * Generate a trajectory starting from a state following a policy.
 */
export function generateTrajectory(
  startState: number,
  policy: Policy,
  config: GridWorldConfig,
  maxSteps: number = 20,
  startAction?: Action
): { state: number; action: Action; reward: number; nextState: number }[] {
  const traj: { state: number; action: Action; reward: number; nextState: number }[] = [];
  let state = startState;

  for (let stepIdx = 0; stepIdx < maxSteps; stepIdx++) {
    if (isTerminal(state, config)) break;
    const action = stepIdx === 0 && startAction !== undefined ? startAction : sampleAction(policy[state]);
    const result = step(state, action, config);
    traj.push({ state, action, reward: result.reward, nextState: result.nextState });
    state = result.nextState;
    if (result.done) break;
  }

  return traj;
}

/**
 * Compute discounted return of a trajectory.
 */
export function discountedReturn(
  trajectory: { reward: number }[],
  gamma: number
): number {
  let g = 0;
  for (let t = trajectory.length - 1; t >= 0; t--) {
    g = trajectory[t].reward + gamma * g;
  }
  return g;
}

/**
 * Compute action-value function q(s,a) given state values v and environment model.
 * q(s,a) = r(s,a) + gamma * v(s')
 */
export function computeQValues(
  values: StateValues,
  config: GridWorldConfig
): number[][] {
  const numStates = config.rows * config.cols;
  const q: number[][] = Array.from({ length: numStates }, () => new Array(5).fill(0));

  for (let s = 0; s < numStates; s++) {
    if (isTerminal(s, config)) {
      q[s] = new Array(5).fill(0);
      continue;
    }

    for (let a = 0; a < 5; a++) {
      const result = step(s, a as Action, config);
      q[s][a] = result.done
        ? result.reward
        : result.reward + config.gamma * values[result.nextState];
    }
  }

  return q;
}

/**
 * Derive a greedy deterministic policy from Q-values.
 */
export function greedyPolicy(qValues: number[][]): Policy {
  return qValues.map((q) => {
    const maxQ = Math.max(...q);
    const bestActions = q
      .map((value, idx) => ({ value, idx }))
      .filter(({ value }) => Math.abs(value - maxQ) < 1e-9)
      .map(({ idx }) => idx);
    const action = bestActions[0];
    const dist = new Array(5).fill(0);
    dist[action] = 1;
    return dist;
  });
}

/**
 * Greedy policy that treats terminal states as absorbing (always stay).
 * Use this for episodic task visualization so that the target cell does not
 * display a random action arrow.
 */
export function greedyPolicyWithTerminal(
  qValues: number[][],
  config: GridWorldConfig
): Policy {
  return qValues.map((q, s) => {
    const dist = new Array(5).fill(0);
    if (isTerminal(s, config)) {
      dist[4] = 1; // stay / terminal
      return dist;
    }
    const maxQ = Math.max(...q);
    const bestActions = q
      .map((value, idx) => ({ value, idx }))
      .filter(({ value }) => Math.abs(value - maxQ) < 1e-9)
      .map(({ idx }) => idx);
    const action = bestActions[0];
    dist[action] = 1;
    return dist;
  });
}

/**
 * Run value iteration and return the history of value functions and policies.
 */
export function valueIteration(
  config: GridWorldConfig,
  maxIterations: number = 100,
  tolerance: number = 1e-6
): { values: StateValues[]; policies: Policy[] } {
  const numStates = config.rows * config.cols;
  let v = new Array(numStates).fill(0);
  const values: StateValues[] = [v];
  const policies: Policy[] = [];

  for (let k = 0; k < maxIterations; k++) {
    const q = computeQValues(v, config);
    const policy = greedyPolicyWithTerminal(q, config);
    policies.push(policy);

    const vNext = q.map((qRow) => Math.max(...qRow));
    values.push(vNext);

    const diff = Math.max(...vNext.map((val, i) => Math.abs(val - v[i])));
    v = vNext;
    if (diff < tolerance) break;
  }

  return { values, policies };
}

/**
 * Run policy iteration and return the history of value functions and policies.
 */
export function policyIteration(
  config: GridWorldConfig,
  maxIterations: number = 100,
  evalIterations: number = 100,
  tolerance: number = 1e-6,
  initialPolicy: Policy | null = null
): { values: StateValues[]; policies: Policy[] } {
  const numStates = config.rows * config.cols;
  const values: StateValues[] = [];
  let policy: Policy = initialPolicy ?? randomPolicy(numStates);
  const policies: Policy[] = [policy];
  let prevValues: StateValues | null = null;

  for (let k = 0; k < maxIterations; k++) {
    const vHistory = iterateStateValues(policy, config, evalIterations, null);
    const v = vHistory[vHistory.length - 1];

    if (prevValues) {
      const diff = Math.max(...v.map((val, i) => Math.abs(val - prevValues![i])));
      if (diff < tolerance) break;
    }
    prevValues = v;
    values.push(v);

    const q = computeQValues(v, config);
    const newPolicy = greedyPolicyWithTerminal(q, config);

    const isSame = policy.every((dist, s) =>
      dist.every((p, a) => Math.abs(p - newPolicy[s][a]) < 1e-9)
    );

    policy = newPolicy;
    policies.push(policy);
    if (isSame) break;
  }

  return { values, policies };
}

/**
 * Apply the Bellman optimality operator once to a value function.
 * Returns (T*v)(s) = max_a q(s,a).
 */
export function bellmanOptimalityOperator(
  values: StateValues,
  config: GridWorldConfig
): StateValues {
  const q = computeQValues(values, config);
  return q.map((row) => Math.max(...row));
}

/**
 * Value iteration with convergence diagnostics.
 * Returns the value/policy history plus per-iteration residuals and errors
 * relative to the optimal value function v*.
 */
export function valueIterationConvergence(
  config: GridWorldConfig,
  maxIterations: number = 200,
  tolerance: number = 1e-6
): {
  values: StateValues[];
  policies: Policy[];
  residuals: number[];
  errors: number[];
} {
  const numStates = config.rows * config.cols;
  const vStar =
    valueIteration(config, 1000, 1e-12).values.at(-1) ??
    new Array(numStates).fill(0);
  let v = new Array(numStates).fill(0);
  const values: StateValues[] = [v];
  const policies: Policy[] = [];
  const residuals: number[] = [];
  const errors: number[] = [];

  for (let k = 0; k < maxIterations; k++) {
    const q = computeQValues(v, config);
    const policy = greedyPolicyWithTerminal(q, config);
    policies.push(policy);
    const vNext = q.map((row) => Math.max(...row));
    values.push(vNext);

    const residual = Math.max(...vNext.map((val, i) => Math.abs(val - v[i])));
    residuals.push(residual);
    errors.push(Math.max(...vNext.map((val, i) => Math.abs(val - vStar[i]))));

    v = vNext;
    if (residual < tolerance) break;
  }

  return { values, policies, residuals, errors };
}

/**
 * Gauss-Seidel (in-place) value iteration.
 * Updates v(s) immediately, so later states in the same sweep use the new values.
 */
export function gaussSeidelValueIteration(
  config: GridWorldConfig,
  maxIterations: number = 200,
  tolerance: number = 1e-6
): { values: StateValues[]; policies: Policy[] } {
  const numStates = config.rows * config.cols;
  let v = new Array(numStates).fill(0);
  const values: StateValues[] = [v];
  const policies: Policy[] = [];

  for (let k = 0; k < maxIterations; k++) {
    let residual = 0;
    for (let s = 0; s < numStates; s++) {
      const old = v[s];
      const q = computeQValues(v, config);
      v[s] = Math.max(...q[s]);
      residual = Math.max(residual, Math.abs(v[s] - old));
    }
    values.push([...v]);
    policies.push(greedyPolicyWithTerminal(computeQValues(v, config), config));
    if (residual < tolerance) break;
  }

  return { values, policies };
}

function shuffleInPlace<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Asynchronous value iteration.
 * Each history entry corresponds to one state update (or one subset update).
 * Modes:
 *   - single-random: one random state per step
 *   - single-sequential: cycle through states in order
 *   - subset-random: a random subset of states per step
 */
export function asyncValueIteration(
  config: GridWorldConfig,
  maxSteps: number = 100,
  tolerance: number = 1e-6,
  mode: 'single-random' | 'single-sequential' | 'subset-random' = 'single-random'
): { values: StateValues[]; policies: Policy[]; updatedStates: number[] } {
  const numStates = config.rows * config.cols;
  let v = new Array(numStates).fill(0);
  const values: StateValues[] = [v];
  const policies: Policy[] = [];
  const updatedStates: number[] = [];

  let seqIndex = 0;
  for (let step = 0; step < maxSteps; step++) {
    let statesToUpdate: number[];
    if (mode === 'single-random') {
      statesToUpdate = [Math.floor(Math.random() * numStates)];
    } else if (mode === 'single-sequential') {
      statesToUpdate = [seqIndex % numStates];
      seqIndex++;
    } else {
      const subsetSize = Math.max(1, Math.ceil(numStates / 3));
      const pool = Array.from({ length: numStates }, (_, i) => i);
      shuffleInPlace(pool);
      statesToUpdate = pool.slice(0, subsetSize);
    }

    let residual = 0;
    for (const s of statesToUpdate) {
      const old = v[s];
      const q = computeQValues(v, config);
      v[s] = Math.max(...q[s]);
      residual = Math.max(residual, Math.abs(v[s] - old));
    }
    updatedStates.push(statesToUpdate[0]);
    values.push([...v]);
    policies.push(greedyPolicyWithTerminal(computeQValues(v, config), config));
    if (residual < tolerance) break;
  }

  return { values, policies, updatedStates };
}

/**
 * Build an epsilon-greedy policy from Q-values.
 */
/**
 * Convert action-value estimates to state values (max over actions).
 */
export function actionValueToStateValue(qValues: number[][]): StateValues {
  return qValues.map((q) => Math.max(...q));
}

/**
 * State values under a given policy: V(s) = Σ_a π(a|s) · Q(s,a)
 */
export function policyWeightedStateValues(
  qValues: number[][],
  policy: Policy
): number[] {
  return qValues.map((qRow, s) =>
    qRow.reduce((sum, qValue, a) => sum + policy[s][a] * qValue, 0)
  );
}

/**
 * Deterministic greedy policy — always picks first maximum action.
 * Useful for stable tie-breaking in policy iteration convergence checks.
 */
export function deterministicGreedyPolicy(
  qValues: number[][],
  config?: GridWorldConfig
): Policy {
  return qValues.map((q, s) => {
    const dist = new Array(q.length).fill(0);
    if (config && isTerminal(s, config)) {
      dist[4] = 1;
      return dist;
    }
    const maxQ = Math.max(...q);
    const bestAction = q.findIndex(
      value => Math.abs(value - maxQ) < 1e-9
    );
    dist[bestAction] = 1;
    return dist;
  });
}

/**
 * Estimate action values for a FIXED policy using MC (first-visit every-episode).
 * This is a single policy-evaluation step, NOT the complete MC Basic control.
 * For each (s,a), run numEpisodes episodes and average returns.
 */
export function estimateActionValuesMC(
  policy: Policy,
  config: GridWorldConfig,
  numEpisodes: number = 50,
  maxSteps: number = 30 // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
): { policy: Policy; qValues: number[][]; returns: number[][][] } {
  const numStates = config.rows * config.cols;
  const returns: number[][][] = Array.from({ length: numStates }, () =>
    Array.from({ length: 5 }, () => [])
  );

  for (let s = 0; s < numStates; s++) {
    for (let a = 0; a < 5; a++) {
      for (let ep = 0; ep < numEpisodes; ep++) {
        const traj = generateTrajectoryFrom(s, a as Action, policy, config, maxSteps);
        const g = discountedReturn(traj, config.gamma);
        returns[s][a].push(g);
      }
    }
  }

  const qValues = returns.map((sReturns) =>
    sReturns.map((aReturns) =>
      aReturns.length > 0
        ? aReturns.reduce((sum, g) => sum + g, 0) / aReturns.length
        : 0
    )
  );

  return { policy, qValues, returns };
}

function generateTrajectoryFrom(
  startState: number,
  startAction: Action,
  policy: Policy,
  config: GridWorldConfig,
  horizonT: number
): { reward: number }[] {
  const traj: { reward: number }[] = [];
  // first transition uses 1 step of the horizon
  const first = step(startState, startAction, config);
  traj.push({ reward: first.reward });
  let state = first.nextState;

  for (let stepIdx = 1; stepIdx < horizonT; stepIdx++) {
    if (isTerminal(state, config)) break;
    const action = sampleAction(policy[state]);
    const result = step(state, action, config);
    traj.push({ reward: result.reward });
    state = result.nextState;
    if (result.done) break;
  }

  return traj;
}

/**
 * Estimate the state value of a start state by Monte Carlo sample averages.
 * Returns the history of estimates after each episode.
 */
export function estimateStateValueMC(
  startState: number,
  policy: Policy,
  config: GridWorldConfig,
  numEpisodes: number,
  maxSteps: number = 30 // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
): { estimates: number[]; returns: number[] } {
  const estimates: number[] = [];
  const returns: number[] = [];
  let sum = 0;

  for (let ep = 0; ep < numEpisodes; ep++) {
    const traj = generateTrajectory(startState, policy, config, maxSteps);
    const g = discountedReturn(traj, config.gamma);
    sum += g;
    returns.push(g);
    estimates.push(sum / (ep + 1));
  }

  return { estimates, returns };
}

export type EpsilonScheduleMode = 'fixed' | 'decay-floor' | 'glie';

export function epsilonAtEpisode(
  episode: number,
  epsilon0: number,
  mode: EpsilonScheduleMode
): number {
  if (mode === 'fixed') return epsilon0;
  if (mode === 'glie') return epsilon0 / Math.sqrt(episode + 1);
  // decay-floor
  return Math.max(0.01, epsilon0 * Math.pow(0.99, episode));
}

function epsilonGreedyDistribution(
  qState: number[],
  epsilon: number,
  rng?: () => number
): number[] {
  const numActions = qState.length;
  const maxQ = Math.max(...qState);
  const bestActions = qState
    .map((value, idx) => ({ value, idx }))
    .filter(({ value }) => Math.abs(value - maxQ) < 1e-9)
    .map(({ idx }) => idx);
  const best = rng
    ? bestActions[Math.floor(rng() * bestActions.length)]
    : bestActions[0];
  const dist = new Array(numActions).fill(epsilon / numActions);
  dist[best] += 1 - epsilon;
  return dist;
}

export function epsilonGreedyPolicy(
  qValues: number[][],
  epsilon: number,
  rng?: () => number
): Policy {
  return qValues.map((q) => epsilonGreedyDistribution(q, epsilon, rng));
}

export interface TDUpdateRecord {
  episode: number;
  time: number;
  state: number;
  action: Action;
  reward: number;
  nextState: number;
  done: boolean;
  oldEstimate: number;
  bootstrapValue: number;
  target: number;
  tdError: number;
  newEstimate: number;
  valuesBefore?: number[];
  valuesAfter?: number[];
  qBefore?: number[][];
  qAfter?: number[][];
  nextAction?: Action;
  behaviorPolicy?: number[];
  behaviorPolicyBefore?: Policy;
  behaviorPolicyAfter?: Policy;
  greedyPolicyBefore?: Policy;
  greedyPolicyAfter?: Policy;
  rewardTerms?: number[];
  bootstrapState?: number;
  bootstrapAction?: Action;
  bootstrapExponent?: number;
  naturalTerminal?: boolean;
  truncated?: boolean;
}

export interface PredictionFrame {
  kind: 'v';
  values: number[];
  policy: Policy;
}

export interface ControlFrame {
  kind: 'q';
  qValues: number[][];
  behaviorPolicy: Policy;
  greedyPolicy: Policy;
}

export type TDHistoryFrame = PredictionFrame | ControlFrame;

export interface PredictionResult {
  frames: PredictionFrame[];
  updates: TDUpdateRecord[];
}

export interface ControlResult {
  frames: ControlFrame[];
  updates: TDUpdateRecord[];
}

function copyV(v: number[]): number[] {
  return v.map((x) => x);
}

function copyQ(q: number[][]): number[][] {
  return q.map((row) => [...row]);
}

export function policyBellmanResidualV(
  values: number[],
  policy: Policy,
  config: GridWorldConfig
): number {
  let maxResidual = 0;
  for (let s = 0; s < values.length; s++) {
    if (isTerminal(s, config)) continue;
    let expected = 0;
    for (let a = 0; a < policy[s].length; a++) {
      const p = policy[s][a];
      if (p === 0) continue;
      const result = step(s, a as Action, config);
      const target =
        result.reward + config.gamma * (result.done ? 0 : values[result.nextState]);
      expected += p * target;
    }
    maxResidual = Math.max(maxResidual, Math.abs(values[s] - expected));
  }
  return maxResidual;
}

export function policyBellmanResidualQ(
  q: number[][],
  policy: Policy,
  config: GridWorldConfig
): number {
  let maxResidual = 0;
  for (let s = 0; s < q.length; s++) {
    if (isTerminal(s, config)) continue;
    for (let a = 0; a < q[s].length; a++) {
      const result = step(s, a as Action, config);
      let expectedNext = 0;
      if (!result.done) {
        expectedNext = q[result.nextState].reduce(
          (sum, qVal, b) => sum + policy[result.nextState][b] * qVal,
          0
        );
      }
      const target = result.reward + config.gamma * expectedNext;
      maxResidual = Math.max(maxResidual, Math.abs(q[s][a] - target));
    }
  }
  return maxResidual;
}

export function optimalBellmanResidualQ(
  q: number[][],
  config: GridWorldConfig
): number {
  let maxResidual = 0;
  for (let s = 0; s < q.length; s++) {
    if (isTerminal(s, config)) continue;
    for (let a = 0; a < q[s].length; a++) {
      const result = step(s, a as Action, config);
      const maxQNext = result.done ? 0 : Math.max(...q[result.nextState]);
      const target = result.reward + config.gamma * maxQNext;
      maxResidual = Math.max(maxResidual, Math.abs(q[s][a] - target));
    }
  }
  return maxResidual;
}

interface NStepTargetInput {
  tau: number;
  n: number;
  T: number;
  rewards: number[];
  states: number[];
  actions: Action[];
  q: number[][];
  gamma: number;
  truncated: boolean;
}

function computeNStepTarget(input: NStepTargetInput): number {
  const { tau, n, T, rewards, states, actions, q, gamma, truncated } = input;
  const end = Math.min(tau + n, T);
  let g = 0;
  for (let i = tau + 1; i <= end; i++) {
    g += Math.pow(gamma, i - tau - 1) * rewards[i];
  }
  if (tau + n < T) {
    g += Math.pow(gamma, n) * q[states[tau + n]][actions[tau + n]];
  } else if (truncated && tau + n === T && actions.length > T) {
    // Artificial horizon: bootstrap from the sampled (S_H, A_H).
    g += Math.pow(gamma, n) * q[states[T]][actions[T]];
  }
  return g;
}

/**
 * TD(0) prediction: estimate state values for a given policy.
 * Returns per-episode frames and every transition-level update record.
 */
export function tdZeroPrediction(
  policy: Policy,
  config: GridWorldConfig,
  alpha: number = 0.1,
  horizonH: number = 30, // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
  episodes: number = 200,
  seed: number = 1
): PredictionResult {
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  let v = new Array(numStates).fill(0);
  const frames: PredictionFrame[] = [{ kind: 'v', values: copyV(v), policy }];
  const updates: TDUpdateRecord[] = [];
  let time = 0;

  for (let ep = 0; ep < episodes; ep++) {
    let state = config.startState;
    for (let t = 0; t < horizonH; t++) {
      if (isTerminal(state, config)) break;
      const action = sampleActionWithRng(policy[state], rng);
      const result = step(state, action, config);
      const old = v[state];
      const bootstrap = result.done ? 0 : v[result.nextState];
      const target = result.reward + config.gamma * bootstrap;
      const tdError = target - old;
      const valuesBefore = copyV(v);
      v[state] += alpha * tdError;
      updates.push({
        episode: ep,
        time,
        state,
        action,
        reward: result.reward,
        nextState: result.nextState,
        done: result.done,
        oldEstimate: old,
        bootstrapValue: bootstrap,
        target,
        tdError,
        newEstimate: v[state],
        valuesBefore,
        valuesAfter: copyV(v),
        behaviorPolicy: policy[state],
        behaviorPolicyBefore: policy,
        behaviorPolicyAfter: policy,
        naturalTerminal: result.done,
        truncated: false,
      });
      state = result.nextState;
      time++;
      if (result.done) break;
    }
    frames.push({ kind: 'v', values: copyV(v), policy });
  }

  return { frames, updates };
}

/**
 * Sarsa: on-policy TD control with ε-greedy exploration.
 */
export function sarsa(
  config: GridWorldConfig,
  alpha: number = 0.1,
  epsilon0: number = 0.3,
  epsilonMode: EpsilonScheduleMode = 'fixed',
  horizonH: number = 30, // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
  episodes: number = 200,
  seed: number = 1
): ControlResult {
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;
  let q = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  const frames: ControlFrame[] = [
    {
      kind: 'q',
      qValues: copyQ(q),
      behaviorPolicy: epsilonGreedyPolicy(q, epsilon0, rng),
      greedyPolicy: greedyPolicy(q),
    },
  ];
  const updates: TDUpdateRecord[] = [];
  let time = 0;

  for (let ep = 0; ep < episodes; ep++) {
    const epsilon = epsilonAtEpisode(ep, epsilon0, epsilonMode);
    let state = config.startState;
    let action = sampleActionWithRng(
      epsilonGreedyDistribution(q[state], epsilon, rng),
      rng
    );

    for (let t = 0; t < horizonH; t++) {
      const result = step(state, action, config);
      let nextAction: Action | undefined;
      let bootstrap = 0;
      if (!result.done) {
        nextAction = sampleActionWithRng(
          epsilonGreedyDistribution(q[result.nextState], epsilon, rng),
          rng
        );
        bootstrap = q[result.nextState][nextAction];
      }
      const target = result.reward + config.gamma * bootstrap;
      const tdError = target - q[state][action];
      const qBefore = copyQ(q);
      const behaviorPolicyBefore = epsilonGreedyPolicy(q, epsilon);
      const greedyPolicyBefore = greedyPolicy(q);
      q[state][action] += alpha * tdError;
      const qAfter = copyQ(q);
      const behaviorPolicyAfter = epsilonGreedyPolicy(qAfter, epsilon);
      const greedyPolicyAfter = greedyPolicy(qAfter);
      updates.push({
        episode: ep,
        time,
        state,
        action,
        reward: result.reward,
        nextState: result.nextState,
        done: result.done,
        oldEstimate: qBefore[state][action],
        bootstrapValue: bootstrap,
        target,
        tdError,
        newEstimate: q[state][action],
        qBefore,
        qAfter,
        nextAction,
        behaviorPolicy: behaviorPolicyBefore[state],
        behaviorPolicyBefore,
        behaviorPolicyAfter,
        greedyPolicyBefore,
        greedyPolicyAfter,
        naturalTerminal: result.done,
        truncated: false,
      });
      if (result.done) break;
      state = result.nextState;
      action = nextAction!;
      time++;
    }
    frames.push({
      kind: 'q',
      qValues: copyQ(q),
      behaviorPolicy: epsilonGreedyPolicy(q, epsilon, rng),
      greedyPolicy: greedyPolicy(q),
    });
  }

  return { frames, updates };
}

/**
 * Q-learning: off-policy TD control.
 */
export function qLearning(
  config: GridWorldConfig,
  alpha: number = 0.1,
  epsilon0: number = 0.3,
  epsilonMode: EpsilonScheduleMode = 'fixed',
  horizonH: number = 30, // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
  episodes: number = 200,
  seed: number = 1
): ControlResult {
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;
  let q = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  const frames: ControlFrame[] = [
    {
      kind: 'q',
      qValues: copyQ(q),
      behaviorPolicy: epsilonGreedyPolicy(q, epsilon0, rng),
      greedyPolicy: greedyPolicy(q),
    },
  ];
  const updates: TDUpdateRecord[] = [];
  let time = 0;

  for (let ep = 0; ep < episodes; ep++) {
    const epsilon = epsilonAtEpisode(ep, epsilon0, epsilonMode);
    let state = config.startState;

    for (let t = 0; t < horizonH; t++) {
      if (isTerminal(state, config)) break;
      const behaviorPolicy = epsilonGreedyPolicy(q, epsilon, rng);
      const action = sampleActionWithRng(behaviorPolicy[state], rng);
      const result = step(state, action, config);
      const maxQNext = result.done ? 0 : Math.max(...q[result.nextState]);
      const target = result.reward + config.gamma * maxQNext;
      const tdError = target - q[state][action];
      const qBefore = copyQ(q);
      const behaviorPolicyBefore = epsilonGreedyPolicy(q, epsilon);
      const greedyPolicyBefore = greedyPolicy(q);
      q[state][action] += alpha * tdError;
      const qAfter = copyQ(q);
      const behaviorPolicyAfter = epsilonGreedyPolicy(qAfter, epsilon);
      const greedyPolicyAfter = greedyPolicy(qAfter);
      updates.push({
        episode: ep,
        time,
        state,
        action,
        reward: result.reward,
        nextState: result.nextState,
        done: result.done,
        oldEstimate: qBefore[state][action],
        bootstrapValue: maxQNext,
        target,
        tdError,
        newEstimate: q[state][action],
        qBefore,
        qAfter,
        behaviorPolicy: behaviorPolicy[state],
        behaviorPolicyBefore,
        behaviorPolicyAfter,
        greedyPolicyBefore,
        greedyPolicyAfter,
        naturalTerminal: result.done,
        truncated: false,
      });
      state = result.nextState;
      time++;
      if (result.done) break;
    }
    frames.push({
      kind: 'q',
      qValues: copyQ(q),
      behaviorPolicy: epsilonGreedyPolicy(q, epsilon, rng),
      greedyPolicy: greedyPolicy(q),
    });
  }

  return { frames, updates };
}

/**
 * n-step Sarsa: on-policy TD control with n-step returns.
 *
 * This is the standard online algorithm: transitions are generated one at a
 * time, and the n-step update is performed as soon as n transitions are
 * available. The trajectory is capped at H transitions; natural terminal and
 * artificial horizon are handled separately so no tail updates are lost.
 */
export function nStepSarsa(
  config: GridWorldConfig,
  alpha: number = 0.1,
  epsilon0: number = 0.3,
  epsilonMode: EpsilonScheduleMode = 'fixed',
  n: number = 3,
  horizonH: number = 30, // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
  episodes: number = 200,
  seed: number = 1
): ControlResult {
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;
  let q = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  const frames: ControlFrame[] = [
    {
      kind: 'q',
      qValues: copyQ(q),
      behaviorPolicy: epsilonGreedyPolicy(q, epsilon0, rng),
      greedyPolicy: greedyPolicy(q),
    },
  ];
  const updates: TDUpdateRecord[] = [];
  let time = 0;

  for (let ep = 0; ep < episodes; ep++) {
    const epsilon = epsilonAtEpisode(ep, epsilon0, epsilonMode);
    const states: number[] = [config.startState];
    const actions: Action[] = [];
    const rewards: number[] = [0];
    let T = Infinity;
    let truncated = false;
    let t = 0;

    // Choose the initial action A_0.
    const startAction = sampleActionWithRng(
      epsilonGreedyDistribution(q[config.startState], epsilon, rng),
      rng
    );
    actions.push(startAction);

    while (true) {
      if (t < T) {
        const result = step(states[t], actions[t], config);
        rewards.push(result.reward);
        states.push(result.nextState);

        if (result.done) {
          T = t + 1;
        } else if (t + 1 === horizonH) {
          // Artificial horizon: the trajectory has reached H transitions.
          // Sample a bootstrap action from S_H for truncated n-step targets.
          const bootstrapAction = sampleActionWithRng(
            epsilonGreedyDistribution(q[result.nextState], epsilon, rng),
            rng
          );
          actions.push(bootstrapAction);
          truncated = true;
          T = horizonH;
        } else {
          // Continue the trajectory by sampling the next action online.
          const nextAction = sampleActionWithRng(
            epsilonGreedyDistribution(q[result.nextState], epsilon, rng),
            rng
          );
          actions.push(nextAction);
        }
      }

      const tau = t - n + 1;
      if (tau >= 0) {
        const target = computeNStepTarget({
          tau,
          n,
          T,
          rewards,
          states,
          actions,
          q,
          gamma: config.gamma,
          truncated,
        });
        const s = states[tau];
        const a = actions[tau];
        const tdError = target - q[s][a];
        const qBefore = copyQ(q);
        const behaviorPolicyBefore = epsilonGreedyPolicy(qBefore, epsilon);
        const greedyPolicyBefore = greedyPolicy(qBefore);
        q[s][a] += alpha * tdError;
        const qAfter = copyQ(q);
        const behaviorPolicyAfter = epsilonGreedyPolicy(qAfter, epsilon);
        const greedyPolicyAfter = greedyPolicy(qAfter);

        const bootstrapValue =
          tau + n < T
            ? qBefore[states[tau + n]][actions[tau + n]]
            : truncated && tau + n === T && actions.length > T
              ? qBefore[states[T]][actions[T]]
              : 0;

        const end = Math.min(tau + n, T);
        const rewardTerms: number[] = [];
        for (let i = tau + 1; i <= end; i++) rewardTerms.push(rewards[i]);
        const bootstrapExponent = rewardTerms.length;
        let bootstrapState: number | undefined;
        let bootstrapAction: Action | undefined;
        if (tau + n < T) {
          bootstrapState = states[tau + n];
          bootstrapAction = actions[tau + n];
        } else if (truncated && tau + n === T && actions.length > T) {
          bootstrapState = states[T];
          bootstrapAction = actions[T];
        }

        updates.push({
          episode: ep,
          time,
          state: s,
          action: a,
          reward: rewards[tau + 1],
          nextState: states[tau + 1],
          done: tau + 1 === T && !truncated,
          oldEstimate: qBefore[s][a],
          bootstrapValue,
          target,
          tdError,
          newEstimate: q[s][a],
          qBefore,
          qAfter,
          behaviorPolicy: behaviorPolicyBefore[s],
          behaviorPolicyBefore,
          behaviorPolicyAfter,
          greedyPolicyBefore,
          greedyPolicyAfter,
          rewardTerms,
          bootstrapState,
          bootstrapAction,
          bootstrapExponent,
          naturalTerminal: tau + 1 === T && !truncated,
          truncated,
        });
        time++;
      }

      if (tau === T - 1) break;
      t++;
    }

    frames.push({
      kind: 'q',
      qValues: copyQ(q),
      behaviorPolicy: epsilonGreedyPolicy(q, epsilon, rng),
      greedyPolicy: greedyPolicy(q),
    });
  }

  return { frames, updates };
}

/**
 * Expected Sarsa: on-policy TD control using the expected value under the
 * current ε-greedy policy as the TD target.
 */
export function expectedSarsa(
  config: GridWorldConfig,
  alpha: number = 0.1,
  epsilon0: number = 0.3,
  epsilonMode: EpsilonScheduleMode = 'fixed',
  horizonH: number = 30, // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
  episodes: number = 200,
  seed: number = 1
): ControlResult {
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;
  let q = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  const frames: ControlFrame[] = [
    {
      kind: 'q',
      qValues: copyQ(q),
      behaviorPolicy: epsilonGreedyPolicy(q, epsilon0, rng),
      greedyPolicy: greedyPolicy(q),
    },
  ];
  const updates: TDUpdateRecord[] = [];
  let time = 0;

  for (let ep = 0; ep < episodes; ep++) {
    const epsilon = epsilonAtEpisode(ep, epsilon0, epsilonMode);
    let state = config.startState;

    for (let t = 0; t < horizonH; t++) {
      if (isTerminal(state, config)) break;
      const behaviorPolicy = epsilonGreedyPolicy(q, epsilon, rng);
      const action = sampleActionWithRng(behaviorPolicy[state], rng);
      const result = step(state, action, config);

      let bootstrap = 0;
      if (!result.done) {
        bootstrap = q[result.nextState].reduce(
          (sum, qVal, b) => sum + behaviorPolicy[result.nextState][b] * qVal,
          0
        );
      }
      const target = result.reward + config.gamma * bootstrap;
      const tdError = target - q[state][action];
      const qBefore = copyQ(q);
      const behaviorPolicyBefore = epsilonGreedyPolicy(q, epsilon);
      const greedyPolicyBefore = greedyPolicy(q);
      q[state][action] += alpha * tdError;
      const qAfter = copyQ(q);
      const behaviorPolicyAfter = epsilonGreedyPolicy(qAfter, epsilon);
      const greedyPolicyAfter = greedyPolicy(qAfter);
      updates.push({
        episode: ep,
        time,
        state,
        action,
        reward: result.reward,
        nextState: result.nextState,
        done: result.done,
        oldEstimate: qBefore[state][action],
        bootstrapValue: bootstrap,
        target,
        tdError,
        newEstimate: q[state][action],
        qBefore,
        qAfter,
        behaviorPolicy: behaviorPolicy[state],
        behaviorPolicyBefore,
        behaviorPolicyAfter,
        greedyPolicyBefore,
        greedyPolicyAfter,
        naturalTerminal: result.done,
        truncated: false,
      });
      state = result.nextState;
      time++;
      if (result.done) break;
    }
    frames.push({
      kind: 'q',
      qValues: copyQ(q),
      behaviorPolicy: epsilonGreedyPolicy(q, epsilon, rng),
      greedyPolicy: greedyPolicy(q),
    });
  }

  return { frames, updates };
}

/**
 * Sarsa(λ) with accumulating eligibility traces.
 */
export function sarsaLambda(
  config: GridWorldConfig,
  alpha: number = 0.1,
  epsilon0: number = 0.3,
  epsilonMode: EpsilonScheduleMode = 'fixed',
  lambda: number = 0.8,
  horizonH: number = 30, // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
  episodes: number = 200,
  seed: number = 1
): ControlResult {
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;
  let q = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  const frames: ControlFrame[] = [
    {
      kind: 'q',
      qValues: copyQ(q),
      behaviorPolicy: epsilonGreedyPolicy(q, epsilon0, rng),
      greedyPolicy: greedyPolicy(q),
    },
  ];
  const updates: TDUpdateRecord[] = [];
  let time = 0;

  for (let ep = 0; ep < episodes; ep++) {
    const epsilon = epsilonAtEpisode(ep, epsilon0, epsilonMode);
    const eligibility = Array.from({ length: numStates }, () =>
      new Array(numActions).fill(0)
    );
    let state = config.startState;
    let action = sampleActionWithRng(
      epsilonGreedyDistribution(q[state], epsilon, rng),
      rng
    );

    for (let t = 0; t < horizonH; t++) {
      const result = step(state, action, config);
      let nextAction: Action | undefined;
      let bootstrap = 0;
      if (!result.done) {
        nextAction = sampleActionWithRng(
          epsilonGreedyDistribution(q[result.nextState], epsilon, rng),
          rng
        );
        bootstrap = q[result.nextState][nextAction];
      }
      const target = result.reward + config.gamma * bootstrap;
      const delta = target - q[state][action];
      const qBefore = copyQ(q);
      const behaviorPolicyBefore = epsilonGreedyPolicy(q, epsilon);
      const greedyPolicyBefore = greedyPolicy(q);

      eligibility[state][action] += 1;
      for (let s = 0; s < numStates; s++) {
        for (let a = 0; a < numActions; a++) {
          q[s][a] += alpha * delta * eligibility[s][a];
          eligibility[s][a] *= config.gamma * lambda;
        }
      }

      const qAfter = copyQ(q);
      const behaviorPolicyAfter = epsilonGreedyPolicy(qAfter, epsilon);
      const greedyPolicyAfter = greedyPolicy(qAfter);

      updates.push({
        episode: ep,
        time,
        state,
        action,
        reward: result.reward,
        nextState: result.nextState,
        done: result.done,
        oldEstimate: qBefore[state][action],
        bootstrapValue: bootstrap,
        target,
        tdError: delta,
        newEstimate: q[state][action],
        qBefore,
        qAfter,
        nextAction,
        behaviorPolicy: behaviorPolicyBefore[state],
        behaviorPolicyBefore,
        behaviorPolicyAfter,
        greedyPolicyBefore,
        greedyPolicyAfter,
        naturalTerminal: result.done,
        truncated: false,
      });

      if (result.done) break;
      state = result.nextState;
      action = nextAction!;
      time++;
    }
    frames.push({
      kind: 'q',
      qValues: copyQ(q),
      behaviorPolicy: epsilonGreedyPolicy(q, epsilon, rng),
      greedyPolicy: greedyPolicy(q),
    });
  }

  return { frames, updates };
}

/**
 * TD(λ) prediction with accumulating eligibility traces for a fixed policy.
 */
export function tdLambdaPrediction(
  policy: Policy,
  config: GridWorldConfig,
  alpha: number = 0.1,
  lambda: number = 0.8,
  horizonH: number = 30, // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
  episodes: number = 200,
  seed: number = 1
): PredictionResult {
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  let v = new Array(numStates).fill(0);
  const frames: PredictionFrame[] = [{ kind: 'v', values: copyV(v), policy }];
  const updates: TDUpdateRecord[] = [];
  let time = 0;

  for (let ep = 0; ep < episodes; ep++) {
    let state = config.startState;
    const eligibility = new Array(numStates).fill(0);

    for (let t = 0; t < horizonH; t++) {
      if (isTerminal(state, config)) break;
      const action = sampleActionWithRng(policy[state], rng);
      const result = step(state, action, config);
      const bootstrap = result.done ? 0 : v[result.nextState];
      const delta = result.reward + config.gamma * bootstrap - v[state];
      const valuesBefore = copyV(v);
      eligibility[state] += 1;
      for (let s = 0; s < numStates; s++) {
        v[s] += alpha * delta * eligibility[s];
        eligibility[s] *= config.gamma * lambda;
      }
      updates.push({
        episode: ep,
        time,
        state,
        action,
        reward: result.reward,
        nextState: result.nextState,
        done: result.done,
        oldEstimate: valuesBefore[state],
        bootstrapValue: bootstrap,
        target: result.reward + config.gamma * bootstrap,
        tdError: delta,
        newEstimate: v[state],
        valuesBefore,
        valuesAfter: copyV(v),
        behaviorPolicy: policy[state],
        behaviorPolicyBefore: policy,
        behaviorPolicyAfter: policy,
        naturalTerminal: result.done,
        truncated: false,
      });
      state = result.nextState;
      time++;
      if (result.done) break;
    }
    frames.push({ kind: 'v', values: copyV(v), policy });
  }

  return { frames, updates };
}

/**
 * REINFORCE on a simple multi-armed bandit.
 * actionRewards[i] is the expected reward of arm i.
 * Returns histories of theta, policy probabilities, and episode rewards.
 */
/**
 * MC Basic: for each (s,a), run numEpisodes episodes and average returns.
 * This is a model-free conversion of policy iteration.
 */
export function mcBasic(
  config: GridWorldConfig,
  numEpisodesPerPair: number = 20,
  maxSteps: number = 30 // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
): { policy: Policy; qValues: number[][]; returns: number[][][] } {
  const numStates = config.rows * config.cols;
  const numActions = 5;
  const returns: number[][][] = Array.from({ length: numStates }, () =>
    Array.from({ length: numActions }, () => [])
  );

  const behaviorPolicy = randomPolicy(numStates, numActions);

  for (let s = 0; s < numStates; s++) {
    for (let a = 0; a < numActions; a++) {
      for (let ep = 0; ep < numEpisodesPerPair; ep++) {
        const traj = generateTrajectory(s, behaviorPolicy, config, maxSteps, a as Action);
        const g = discountedReturn(traj, config.gamma);
        returns[s][a].push(g);
      }
    }
  }

  const qValues = returns.map((sReturns) =>
    sReturns.map((aReturns) =>
      aReturns.length > 0
        ? aReturns.reduce((sum, g) => sum + g, 0) / aReturns.length
        : 0
    )
  );

  return { policy: behaviorPolicy, qValues, returns };
}

/**
 * MC Basic policy iteration — the complete algorithm.
 *
 * for k = 0, 1, 2, ...:
 *   1. For each (s,a), sample episodes starting from (s,a), then follow π_k;
 *      estimate q_{π_k}(s,a) by averaging returns.
 *   2. π_{k+1}(s) = greedy(q_k(s,*))
 *   3. Stop if policy is stable.
 *
 * This is NOT a single policy-evaluation step. The old `mcBasic()` is a
 * single-evaluation helper and has been kept for backward compatibility.
 */
export interface MCBasicIteration {
  iteration: number;
  policyBefore: Policy;
  qEstimate: number[][];
  policyAfter: Policy;
  policyStable: boolean;
  changedStateCount: number;
}

export function mcBasicPolicyIteration(
  config: GridWorldConfig,
  episodesPerPair: number = 20,
  maxSteps: number = 30, // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
  maxPolicyIterations: number = 20,
  initialPolicy?: Policy
): {
  iterations: MCBasicIteration[];
  finalPolicy: Policy;
  finalQ: number[][];
  finalQPolicy: Policy;
} {
  const numStates = config.rows * config.cols;
  const numActions = 5;
  let policy: Policy = initialPolicy ?? randomPolicy(numStates, numActions);
  const iterations: MCBasicIteration[] = [];

  for (let k = 0; k < maxPolicyIterations; k++) {
    // Step 1: MC policy evaluation of current policy
    const { qValues } = estimateActionValuesMC(policy, config, episodesPerPair, maxSteps);

    // Step 2: deterministic greedy policy improvement
    const newPolicy = deterministicGreedyPolicy(qValues, config);

    // Step 3: check stability (deterministic tie-breaking)
    const policyStable = policy.every((dist, s) =>
      dist.every((p, a) => Math.abs(p - newPolicy[s][a]) < 1e-9)
    );

    const changedStateCount = policy.reduce((count, dist, s) => {
      const changed = !dist.every((p, a) => Math.abs(p - newPolicy[s][a]) < 1e-9);
      return count + (changed ? 1 : 0);
    }, 0);

    iterations.push({
      iteration: k,
      policyBefore: policy.map((row) => [...row]),
      qEstimate: qValues.map((row) => [...row]),
      policyAfter: newPolicy.map((row) => [...row]),
      policyStable,
      changedStateCount,
    });

    policy = newPolicy;
    if (policyStable) break;
  }

  const finalQ = iterations.length > 0
    ? iterations[iterations.length - 1].qEstimate
    : Array.from({ length: numStates }, () => new Array(numActions).fill(0));

  // finalQ estimates q_{π_k} where π_k = last policyBefore
  const finalQPolicy = iterations.length > 0
    ? iterations[iterations.length - 1].policyBefore
    : policy;

  return { iterations, finalPolicy: policy, finalQ, finalQPolicy };
}

/**
 * Persistent learner state for incremental MC training.
 * Allows "run N more episodes" to continue from where the previous run left off.
 */
export type TrajectoryStep = {
  state: number;
  action: Action;
  reward: number;
  nextState: number;
};

export interface MCLearnerState {
  q: number[][];
  returnsSum: number[][];
  visitCount: number[][];
  policy: Policy;
  episodesCompleted: number;
  currentEpsilon: number;
  lastTrajectory: TrajectoryStep[];
}

export function createMCLearnerState(config: GridWorldConfig, initialEpsilon: number = 0): MCLearnerState {
  const numStates = config.rows * config.cols;
  const numActions = 5;
  return {
    q: Array.from({ length: numStates }, () => new Array(numActions).fill(0)),
    returnsSum: Array.from({ length: numStates }, () => new Array(numActions).fill(0)),
    visitCount: Array.from({ length: numStates }, () => new Array(numActions).fill(0)),
    policy: randomPolicy(numStates, numActions),
    episodesCompleted: 0,
    currentEpsilon: initialEpsilon,
    lastTrajectory: [],
  };
}

/**
 * Incremental MC Exploring Starts: continues training from learnerState.
 * Does NOT reinitialize Q, returns, or counts.
 */
export function runMCExploringStartsEpisodes(
  learnerState: MCLearnerState,
  config: GridWorldConfig,
  additionalEpisodes: number,
  maxSteps: number = 30, // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
  visitMode: 'first-visit' | 'every-visit' = 'first-visit'
): MCLearnerState {
  const numStates = config.rows * config.cols;
  const numActions = 5;
  // Deep-copy the incoming state so the caller's reference is not mutated
  const q = learnerState.q.map((row) => [...row]);
  const returnsSum = learnerState.returnsSum.map((row) => [...row]);
  const visitCount = learnerState.visitCount.map((row) => [...row]);
  let episodesCompleted = learnerState.episodesCompleted;
  let lastTrajectory = learnerState.lastTrajectory;

  for (let ep = 0; ep < additionalEpisodes; ep++) {
    episodesCompleted++;
    const startState = Math.floor(Math.random() * numStates);
    const startAction = Math.floor(Math.random() * numActions) as Action;
    const policy = greedyPolicy(q);
    const traj = generateTrajectory(startState, policy, config, maxSteps, startAction);
    lastTrajectory = traj;

    const visited = new Set<string>();
    for (let t = 0; t < traj.length; t++) {
      const s = traj[t].state;
      const a = traj[t].action;
      const key = `${s},${a}`;
      if (visitMode === 'first-visit' && visited.has(key)) continue;
      visited.add(key);

      const g = discountedReturn(traj.slice(t), config.gamma);
      returnsSum[s][a] += g;
      visitCount[s][a] += 1;
      q[s][a] = returnsSum[s][a] / visitCount[s][a];
    }
  }

  return {
    q,
    returnsSum,
    visitCount,
    policy: greedyPolicy(q),
    episodesCompleted,
    currentEpsilon: 0,
    lastTrajectory,
  };
}

/**
 * Incremental MC ε-Greedy: continues training from learnerState.
 * Does NOT reinitialize Q, returns, or counts.
 * No exploring starts — all actions sampled from ε-greedy policy.
 */
export function runMCEpsilonGreedyEpisodes(
  learnerState: MCLearnerState,
  config: GridWorldConfig,
  additionalEpisodes: number,
  maxSteps: number = 30, // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
  epsilonSchedule: EpsilonSchedule = 'fixed',
  baseEpsilon: number = 0.3,
  visitMode: 'first-visit' | 'every-visit' = 'first-visit'
): MCLearnerState {
  const q = learnerState.q.map((row) => [...row]);
  const returnsSum = learnerState.returnsSum.map((row) => [...row]);
  const visitCount = learnerState.visitCount.map((row) => [...row]);
  let episodesCompleted = learnerState.episodesCompleted;
  let currentEpsilon = learnerState.currentEpsilon;
  let lastTrajectory = learnerState.lastTrajectory;

  for (let ep = 0; ep < additionalEpisodes; ep++) {
    currentEpsilon = computeEpsilon(epsilonSchedule, baseEpsilon, episodesCompleted);
    episodesCompleted++;

    // No exploring starts: start from config.startState
    const startState = config.startState;
    const policy = epsilonGreedyPolicy(q, currentEpsilon);
    const traj = generateTrajectory(startState, policy, config, maxSteps);
    lastTrajectory = traj;

    const visited = new Set<string>();
    for (let t = 0; t < traj.length; t++) {
      const s = traj[t].state;
      const a = traj[t].action;
      const key = `${s},${a}`;
      if (visitMode === 'first-visit' && visited.has(key)) continue;
      visited.add(key);

      const g = discountedReturn(traj.slice(t), config.gamma);
      returnsSum[s][a] += g;
      visitCount[s][a] += 1;
      q[s][a] = returnsSum[s][a] / visitCount[s][a];
    }
  }

  return {
    q,
    returnsSum,
    visitCount,
    policy: epsilonGreedyPolicy(q, currentEpsilon),
    episodesCompleted,
    currentEpsilon,
    lastTrajectory,
  };
}

/**
 * Estimate the optimal action-value function q* for reference and learning curves.
 */
export function estimateTrueActionValues(config: GridWorldConfig): number[][] {
  const vStar =
    valueIteration(config, 1000, 1e-12).values.at(-1) ??
    new Array(config.rows * config.cols).fill(0);
  return computeQValues(vStar, config);
}

/**
 * Root-mean-square error between two Q-tables.
 */
export function qTableRMSE(q: number[][], qRef: number[][]): number {
  let sum = 0;
  let count = 0;
  for (let s = 0; s < q.length; s++) {
    for (let a = 0; a < q[s].length; a++) {
      sum += Math.pow(q[s][a] - qRef[s][a], 2);
      count++;
    }
  }
  return count === 0 ? 0 : Math.sqrt(sum / count);
}

/**
 * Greedy action agreement: for each non-terminal state with a non-trivial decision,
 * check whether at least one greedy action under `q` is also optimal under `qStar`.
 */
export function greedyActionAgreement(
  q: number[][],
  qStar: number[][],
  config: GridWorldConfig
): { agreement: number; evaluatedStateCount: number } {
  const tolerance = 1e-6;
  let match = 0;
  let evaluated = 0;
  for (let s = 0; s < q.length; s++) {
    if (isTerminal(s, config)) continue;
    const maxQ = Math.max(...q[s]);
    const greedyActions = q[s]
      .map((val, a) => ({ val, a }))
      .filter(({ val }) => Math.abs(val - maxQ) <= tolerance)
      .map(({ a }) => a);
    const maxQStar = Math.max(...qStar[s]);
    const optimalActions = qStar[s]
      .map((val, a) => ({ val, a }))
      .filter(({ val }) => Math.abs(val - maxQStar) <= tolerance)
      .map(({ a }) => a);
    // Exclude states where every action is optimal (no decision meaning).
    if (optimalActions.length === q[s].length) continue;
    evaluated++;
    const optimalSet = new Set(optimalActions);
    if (greedyActions.some((a) => optimalSet.has(a))) match++;
  }
  return { agreement: evaluated === 0 ? 0 : match / evaluated, evaluatedStateCount: evaluated };
}

export type EpsilonSchedule = 'fixed' | 'decaying-with-floor' | 'glie';

export function computeEpsilon(
  schedule: EpsilonSchedule,
  baseEpsilon: number,
  episodeIndex: number,
  epsilonMin: number = 0.05
): number {
  if (schedule === 'fixed') return baseEpsilon;
  if (schedule === 'decaying-with-floor') {
    return Math.max(epsilonMin, baseEpsilon / Math.sqrt(episodeIndex + 1));
  }
  // glie: epsilon_k = epsilon0 / sqrt(k + 1) -> 0
  return baseEpsilon / Math.sqrt(episodeIndex + 1);
}

function computeImportanceSamplingRatio(
  trajectory: { state: number; action: Action }[],
  targetPolicy: Policy,
  behaviorPolicy: Policy,
  startIndex = 0
): number {
  let rho = 1;
  for (let i = startIndex; i < trajectory.length; i++) {
    const { state, action } = trajectory[i];
    const bProb = behaviorPolicy[state][action];
    const piProb = targetPolicy[state][action];
    if (bProb === 0) return 0;
    rho *= piProb / bProb;
  }
  return rho;
}

/**
 * Off-policy MC evaluation for a deterministic target policy using a behavior policy.
 * For each (s,a), episodes start at (s,a) and follow the behavior policy.
 * Supports ordinary and weighted importance sampling.
 */
export function offPolicyMCEvaluation(
  targetPolicy: Policy,
  behaviorPolicy: Policy,
  config: GridWorldConfig,
  numEpisodesPerPair: number = 20,
  type: 'ordinary' | 'weighted' = 'ordinary',
  maxSteps: number = 30 // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
): {
  qValues: number[][];
  qHistory: number[][][];
  lastTrajectory: TrajectoryStep[];
  lastRho: number;
} {
  const numStates = config.rows * config.cols;
  const numActions = 5;
  const qValues: number[][] = Array.from({ length: numStates }, () =>
    new Array(numActions).fill(0)
  );
  const qHistory: number[][][] = [qValues.map((row) => [...row])];

  let lastTrajectory: { state: number; action: Action; reward: number; nextState: number }[] = [];
  let lastRho = 0;

  for (let s = 0; s < numStates; s++) {
    for (let a = 0; a < numActions; a++) {
      if (type === 'ordinary') {
        const estimates: number[] = [];
        for (let ep = 0; ep < numEpisodesPerPair; ep++) {
          const traj = generateTrajectory(s, behaviorPolicy, config, maxSteps, a as Action);
          if (s === 0 && a === 0 && ep === 0) lastTrajectory = traj;
          const rho = computeImportanceSamplingRatio(traj, targetPolicy, behaviorPolicy, 1);
          if (s === 0 && a === 0 && ep === 0) lastRho = rho;
          const g = discountedReturn(traj, config.gamma);
          estimates.push(rho * g);
        }
        qValues[s][a] =
          estimates.length > 0
            ? estimates.reduce((sum, x) => sum + x, 0) / estimates.length
            : 0;
      } else {
        let numerator = 0;
        let denominator = 0;
        for (let ep = 0; ep < numEpisodesPerPair; ep++) {
          const traj = generateTrajectory(s, behaviorPolicy, config, maxSteps, a as Action);
          if (s === 0 && a === 0 && ep === 0) lastTrajectory = traj;
          const rho = computeImportanceSamplingRatio(traj, targetPolicy, behaviorPolicy, 1);
          if (s === 0 && a === 0 && ep === 0) lastRho = rho;
          const g = discountedReturn(traj, config.gamma);
          numerator += rho * g;
          denominator += rho;
        }
        qValues[s][a] = denominator > 0 ? numerator / denominator : 0;
      }
      qHistory.push(qValues.map((row) => [...row]));
    }
  }

  return { qValues, qHistory, lastTrajectory, lastRho };
}

/**
 * Environment validation cases.
 * Returns an array of test results that can be displayed in the UI or logged.
 */
export interface ValidationCase {
  name: string;
  state: number;
  action: Action;
  taskType: TaskType;
  expectedNextState: number;
  expectedReward: number;
  expectedDone: boolean;
  passed: boolean;
  actual: StepResult;
}

type ValidationCaseInput = Omit<ValidationCase, 'passed' | 'actual'>;

export function validateEnvironment(config: GridWorldConfig = DEFAULT_CONFIG): ValidationCase[] {
  const cases: ValidationCaseInput[] = [
    {
      name: 's1 + upward => s1, boundaryReward',
      state: 0,
      action: 0,
      taskType: 'continuing',
      expectedNextState: 0,
      expectedReward: config.boundaryReward,
      expectedDone: false,
    },
    {
      name: 's1 + leftward => s1, boundaryReward',
      state: 0,
      action: 3,
      taskType: 'continuing',
      expectedNextState: 0,
      expectedReward: config.boundaryReward,
      expectedDone: false,
    },
    {
      name: 's5 + rightward => s6, forbiddenReward',
      state: 4,
      action: 1,
      taskType: 'continuing',
      expectedNextState: 5,
      expectedReward: config.forbiddenReward,
      expectedDone: false,
    },
    {
      name: 's8 + rightward => s9, targetReward',
      state: 7,
      action: 1,
      taskType: 'continuing',
      expectedNextState: 8,
      expectedReward: config.targetReward,
      expectedDone: false,
    },
    {
      name: 'continuing: s9 + stay => s9, targetReward, done=false',
      state: 8,
      action: 4,
      taskType: 'continuing',
      expectedNextState: 8,
      expectedReward: config.targetReward,
      expectedDone: false,
    },
    {
      name: 'continuing: s9 + rightward => s9, boundaryReward, done=false',
      state: 8,
      action: 1,
      taskType: 'continuing',
      expectedNextState: 8,
      expectedReward: config.boundaryReward,
      expectedDone: false,
    },
    {
      name: 'episodic: entering s9 => done=true',
      state: 7,
      action: 1,
      taskType: 'episodic',
      expectedNextState: 8,
      expectedReward: config.targetReward,
      expectedDone: true,
    },
  ];

  return cases.map((c) => {
    const testConfig = { ...config, taskType: c.taskType };
    const actual = step(c.state, c.action, testConfig);
    const passed =
      actual.nextState === c.expectedNextState &&
      actual.reward === c.expectedReward &&
      actual.done === c.expectedDone;
    return { ...c, actual, passed };
  });
}

