/**
 * GridWorld environment and RL utilities.
 *
 * The book uses a 3x3 grid world as the running example for Chapter 1-2:
 *   s1 s2 s3
 *   s4 s5 s6
 *   s7 s8 s9
 *
 * Default layout:
 *   - Start:    s1
 *   - Target:   s9  (reward +1)
 *   - Forbidden: s3, s5 (reward -1)
 *   - Boundary collision: reward -1, stay in place
 *   - Step: reward 0
 */

export type Action = 0 | 1 | 2 | 3 | 4;
export type StateIdx = number;
export const ACTION_NAMES = ['上', '右', '下', '左', '停留'] as const;
export const ACTION_DELTAS: { dr: number; dc: number }[] = [
  { dr: -1, dc: 0 }, // up
  { dr: 0, dc: 1 },  // right
  { dr: 1, dc: 0 },  // down
  { dr: 0, dc: -1 }, // left
  { dr: 0, dc: 0 },  // stay
];

export type Policy = number[][]; // policy[s][a] = probability
export type StateValues = number[];

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
}

export const DEFAULT_CONFIG: GridWorldConfig = {
  rows: 3,
  cols: 3,
  startState: 0,
  targetState: 8,
  forbiddenStates: [2, 4],
  targetReward: 1,
  forbiddenReward: -1,
  boundaryReward: -1,
  stepReward: 0,
  gamma: 0.9,
};

/** GridWorld with mild boundary penalty (for illustrating policy factors). */
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

/** GridWorld with high discount factor (far-sighted policies). */
export const FAR_SIGHTED_CONFIG: GridWorldConfig = {
  ...DEFAULT_CONFIG,
  gamma: 0.99,
};

/** GridWorld with low discount factor (short-sighted policies). */
export const SHORT_SIGHTED_CONFIG: GridWorldConfig = {
  ...DEFAULT_CONFIG,
  gamma: 0.3,
};

export function stateToRowCol(state: number, cols: number): { row: number; col: number } {
  return { row: Math.floor(state / cols), col: state % cols };
}

export function rowColToState(row: number, col: number, cols: number): number {
  return row * cols + col;
}

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

export function reward(state: number, config: GridWorldConfig): number {
  if (state === config.targetState) return config.targetReward;
  if (config.forbiddenStates.includes(state)) return config.forbiddenReward;
  return config.stepReward;
}

export function isTerminal(state: number, config: GridWorldConfig): boolean {
  return state === config.targetState;
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
 * Compute the Bellman equation for a given policy:
 *   v = r_pi + gamma * P_pi * v
 * where r_pi[s] = sum_a pi(a|s) * r(s,a)
 * and   P_pi[s,s'] = sum_a pi(a|s) * p(s'|s,a).
 *
 * For our deterministic grid world, p(s'|s,a) = 1 iff s' = nextState(s,a).
 */
export function computeBellmanComponents(
  policy: Policy,
  config: GridWorldConfig
): { rPi: number[]; pPi: number[][] } {
  const numStates = config.rows * config.cols;
  const rPi: number[] = new Array(numStates).fill(0);
  const pPi: number[][] = Array.from({ length: numStates }, () => new Array(numStates).fill(0));

  for (let s = 0; s < numStates; s++) {
    for (let a = 0; a < 5; a++) {
      const prob = policy[s][a];
      if (prob === 0) continue;
      const sNext = nextState(s, a as Action, config);
      const r = reward(sNext, config);
      rPi[s] += prob * r;
      pPi[s][sNext] += prob;
    }
  }

  return { rPi, pPi };
}

/**
 * Solve state values exactly: v = (I - gamma * P_pi)^{-1} * r_pi.
 */
export function solveStateValues(policy: Policy, config: GridWorldConfig): StateValues {
  const numStates = config.rows * config.cols;
  const { rPi, pPi } = computeBellmanComponents(policy, config);

  // Build A = I - gamma * P_pi
  const A: number[][] = Array.from({ length: numStates }, (_, i) =>
    Array.from({ length: numStates }, (_, j) => (i === j ? 1 : 0) - config.gamma * pPi[i][j])
  );

  // Gaussian elimination to solve A x = b
  return solveLinearSystem(A, rPi);
}

function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M: number[][] = A.map((row, i) => [...row, b[i]]);

  for (let i = 0; i < n; i++) {
    // Partial pivoting
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

  for (let step = 0; step < maxSteps; step++) {
    if (isTerminal(state, config)) break;
    const action = step === 0 && startAction !== undefined ? startAction : sampleAction(policy[state]);
    const sNext = nextState(state, action, config);
    const r = reward(sNext, config);
    traj.push({ state, action, reward: r, nextState: sNext });
    state = sNext;
    if (isTerminal(state, config)) break;
  }

  return traj;
}

function sampleAction(probs: number[]): Action {
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r <= cum) return i as Action;
  }
  return (probs.length - 1) as Action;
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
 * q(s,a) = sum_{s'} p(s'|s,a) [r(s') + gamma * v(s')]
 */
export function computeQValues(
  values: StateValues,
  config: GridWorldConfig
): number[][] {
  const numStates = config.rows * config.cols;
  const q: number[][] = Array.from({ length: numStates }, () => new Array(5).fill(0));

  for (let s = 0; s < numStates; s++) {
    for (let a = 0; a < 5; a++) {
      const sNext = nextState(s, a as Action, config);
      const r = reward(sNext, config);
      q[s][a] = r + config.gamma * values[sNext];
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
    const action = bestActions[Math.floor(Math.random() * bestActions.length)];
    const dist = new Array(5).fill(0);
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
    const policy = greedyPolicy(q);
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
    // Policy evaluation (iterative, so we can also do truncated evaluation)
    const vHistory = iterateStateValues(policy, config, evalIterations, null);
    const v = vHistory[vHistory.length - 1];

    if (prevValues) {
      const diff = Math.max(...v.map((val, i) => Math.abs(val - prevValues![i])));
      if (diff < tolerance) break;
    }
    prevValues = v;
    values.push(v);

    // Policy improvement
    const q = computeQValues(v, config);
    const newPolicy = greedyPolicy(q);

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
 * Build an epsilon-greedy policy from Q-values.
 */
export function epsilonGreedyPolicy(qValues: number[][], epsilon: number): Policy {
  const numActions = qValues[0]?.length ?? 5;
  return qValues.map((q) => {
    const maxQ = Math.max(...q);
    const bestActions = q
      .map((value, idx) => ({ value, idx }))
      .filter(({ value }) => Math.abs(value - maxQ) < 1e-9)
      .map(({ idx }) => idx);
    const best = bestActions[Math.floor(Math.random() * bestActions.length)];
    const dist = new Array(numActions).fill(epsilon / numActions);
    dist[best] += 1 - epsilon;
    return dist;
  });
}

/**
 * Convert action-value estimates to state values (max over actions).
 */
export function actionValueToStateValue(qValues: number[][]): StateValues {
  return qValues.map((q) => Math.max(...q));
}

/**
 * Estimate action values using MC Basic (first-visit every-episode).
 * For each (s,a), run numEpisodes episodes and average returns.
 */
export function estimateActionValuesMC(
  policy: Policy,
  config: GridWorldConfig,
  numEpisodes: number = 50,
  maxSteps: number = 30
): { qValues: number[][]; returns: number[][][] } {
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

  return { qValues, returns };
}

function generateTrajectoryFrom(
  startState: number,
  startAction: Action,
  policy: Policy,
  config: GridWorldConfig,
  maxSteps: number
): { reward: number }[] {
  const traj: { reward: number }[] = [];
  let state = nextState(startState, startAction, config);
  let r = reward(state, config);
  traj.push({ reward: r });

  for (let step = 0; step < maxSteps; step++) {
    if (isTerminal(state, config)) break;
    const action = sampleAction(policy[state]);
    const sNext = nextState(state, action, config);
    r = reward(sNext, config);
    traj.push({ reward: r });
    state = sNext;
    if (isTerminal(state, config)) break;
  }

  return traj;
}


/**
 * TD(0) prediction: estimate state values for a given policy.
 * Returns the history of value estimates (one entry per episode).
 */
export function tdZeroPrediction(
  policy: Policy,
  config: GridWorldConfig,
  alpha: number = 0.1,
  episodes: number = 200,
  maxSteps: number = 30
): StateValues[] {
  const numStates = config.rows * config.cols;
  let v = new Array(numStates).fill(0);
  const history: StateValues[] = [v.map((x) => x)];

  for (let ep = 0; ep < episodes; ep++) {
    let state = config.startState;
    for (let step = 0; step < maxSteps; step++) {
      if (isTerminal(state, config)) break;
      const action = sampleAction(policy[state]);
      const sNext = nextState(state, action, config);
      const r = reward(sNext, config);
      const tdTarget = r + config.gamma * v[sNext];
      v[state] += alpha * (tdTarget - v[state]);
      state = sNext;
    }
    history.push(v.map((x) => x));
  }

  return history;
}

/**
 * Sarsa: on-policy TD control.
 * Returns the history of Q-value estimates (one entry per episode).
 */
export function sarsa(
  config: GridWorldConfig,
  alpha: number = 0.1,
  epsilon: number = 0.3,
  episodes: number = 200,
  maxSteps: number = 30
): number[][][] {
  const numStates = config.rows * config.cols;
  const numActions = 5;
  let q = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  const history: number[][][] = [q.map((row) => [...row])];

  for (let ep = 0; ep < episodes; ep++) {
    let state = config.startState;
    let policy = epsilonGreedyPolicy(q, epsilon);
    let action = sampleAction(policy[state]);

    for (let step = 0; step < maxSteps; step++) {
      if (isTerminal(state, config)) break;
      const sNext = nextState(state, action, config);
      const r = reward(sNext, config);
      policy = epsilonGreedyPolicy(q, epsilon);
      const aNext = sampleAction(policy[sNext]);
      const tdTarget = r + config.gamma * q[sNext][aNext];
      q[state][action] += alpha * (tdTarget - q[state][action]);
      state = sNext;
      action = aNext;
    }
    history.push(q.map((row) => [...row]));
  }

  return history;
}

/**
 * Q-learning: off-policy TD control.
 * Returns the history of Q-value estimates (one entry per episode).
 */
export function qLearning(
  config: GridWorldConfig,
  alpha: number = 0.1,
  epsilon: number = 0.3,
  episodes: number = 200,
  maxSteps: number = 30
): number[][][] {
  const numStates = config.rows * config.cols;
  const numActions = 5;
  let q = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  const history: number[][][] = [q.map((row) => [...row])];

  for (let ep = 0; ep < episodes; ep++) {
    let state = config.startState;
    for (let step = 0; step < maxSteps; step++) {
      if (isTerminal(state, config)) break;
      const policy = epsilonGreedyPolicy(q, epsilon);
      const action = sampleAction(policy[state]);
      const sNext = nextState(state, action, config);
      const r = reward(sNext, config);
      const maxQNext = Math.max(...q[sNext]);
      const tdTarget = r + config.gamma * maxQNext;
      q[state][action] += alpha * (tdTarget - q[state][action]);
      state = sNext;
    }
    history.push(q.map((row) => [...row]));
  }

  return history;
}

/**
 * Stochastic gradient descent on a quadratic loss f(w) = (w - wTrue)^2.
 * Returns the history of parameter estimates.
 */
export function sgdQuadratic(
  wTrue: number,
  initialW: number,
  alpha: number,
  noiseStd: number,
  iterations: number
): number[] {
  const history: number[] = [initialW];
  let w = initialW;
  for (let k = 0; k < iterations; k++) {
    const noise = (Math.random() * 2 - 1) * noiseStd;
    const gradient = 2 * (w - wTrue) + noise;
    w -= alpha * gradient;
    history.push(w);
  }
  return history;
}


/**
 * Softmax policy from action preferences.
 */
function softmaxPolicy(preferences: number[]): number[] {
  const maxPref = Math.max(...preferences);
  const exps = preferences.map((p) => Math.exp(p - maxPref));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/**
 * REINFORCE on a simple multi-armed bandit.
 * actionRewards[i] is the expected reward of arm i.
 * Returns histories of theta, policy probabilities, and episode rewards.
 */
export function reinforceBandit(
  actionRewards: number[],
  initialTheta: number[],
  alpha: number,
  episodes: number
): {
  thetaHistory: number[][];
  policyHistory: number[][];
  rewardHistory: number[];
} {
  const theta = [...initialTheta];
  const thetaHistory: number[][] = [[...theta]];
  const policyHistory: number[][] = [softmaxPolicy(theta)];
  const rewardHistory: number[] = [];

  for (let ep = 0; ep < episodes; ep++) {
    const policy = softmaxPolicy(theta);
    const action = sampleAction(policy);
    // Deterministic expected reward for this simple demo
    const r = actionRewards[action] + (Math.random() * 2 - 1) * 0.5;
    rewardHistory.push(r);

    for (let a = 0; a < theta.length; a++) {
      const indicator = a === action ? 1 : 0;
      theta[a] += alpha * r * (indicator - policy[a]);
    }

    thetaHistory.push([...theta]);
    policyHistory.push(softmaxPolicy(theta));
  }

  return { thetaHistory, policyHistory, rewardHistory };
}

/**
 * Actor-Critic for the GridWorld.
 * Actor uses softmax action preferences H[s][a].
 * Critic estimates state values v[s].
 * Returns histories of values, policies, and episode total rewards.
 */
export function actorCritic(
  config: GridWorldConfig,
  actorAlpha: number = 0.05,
  criticAlpha: number = 0.1,
  episodes: number = 200,
  maxSteps: number = 30
): { values: StateValues[]; policies: Policy[]; rewardHistory: number[] } {
  const numStates = config.rows * config.cols;
  const numActions = 5;
  let h = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  let v = new Array(numStates).fill(0);

  const values: StateValues[] = [v.map((x) => x)];
  const policies: Policy[] = [h.map((row) => softmaxPolicy(row))];
  const rewardHistory: number[] = [0];

  for (let ep = 0; ep < episodes; ep++) {
    let state = config.startState;
    let episodeReward = 0;

    for (let step = 0; step < maxSteps; step++) {
      if (isTerminal(state, config)) break;
      const policy = softmaxPolicy(h[state]);
      const action = sampleAction(policy);
      const sNext = nextState(state, action as Action, config);
      const r = reward(sNext, config);
      episodeReward += r;

      const tdError = r + config.gamma * v[sNext] - v[state];

      // Critic update
      v[state] += criticAlpha * tdError;

      // Actor update
      for (let a = 0; a < numActions; a++) {
        const indicator = a === action ? 1 : 0;
        h[state][a] += actorAlpha * tdError * (indicator - policy[a]);
      }

      state = sNext;
    }

    values.push(v.map((x) => x));
    policies.push(h.map((row) => softmaxPolicy(row)));
    rewardHistory.push(episodeReward);
  }

  return { values, policies, rewardHistory };
}


/**
 * Mean estimation by sample average (motivating example for MC and SA).
 * Returns the history of estimates.
 */
export function sampleMeanEstimation(
  trueMean: number,
  initialEstimate: number,
  samples: number
): { estimates: number[]; samples: number[] } {
  let estimate = initialEstimate;
  const estimates: number[] = [estimate];
  const sampleHistory: number[] = [];

  for (let i = 0; i < samples; i++) {
    const x = trueMean + (Math.random() * 2 - 1);
    sampleHistory.push(x);
    estimate += (x - estimate) / (i + 1);
    estimates.push(estimate);
  }

  return { estimates, samples: sampleHistory };
}

/**
 * Robbins-Monro algorithm for finding root of g(w) = E[f(w, eta)] = 0.
 * Here we solve g(w) = w - wTrue = 0 with noisy observations.
 * stepSizes is an array of alpha_k.
 */
export function robbinsMonro(
  wTrue: number,
  initialW: number,
  stepSizes: number[]
): number[] {
  const history: number[] = [initialW];
  let w = initialW;

  for (const alpha of stepSizes) {
    const noise = (Math.random() * 2 - 1) * 0.5;
    const observation = w - wTrue + noise;
    w -= alpha * observation;
    history.push(w);
  }

  return history;
}

/**
 * Dvoretzky-style mean estimation: w_{k+1} = w_k + alpha_k (x_k - w_k).
 * Returns the history of estimates.
 */
export function dvoretzkyMeanEstimation(
  trueMean: number,
  initialW: number,
  stepSizes: number[]
): number[] {
  const history: number[] = [initialW];
  let w = initialW;

  for (const alpha of stepSizes) {
    const x = trueMean + (Math.random() * 2 - 1);
    w += alpha * (x - w);
    history.push(w);
  }

  return history;
}

/**
 * Generate step sizes alpha_k = 1 / k^power, optionally shifted.
 */
export function powerStepSizes(
  n: number,
  power: number = 1,
  offset: number = 1
): number[] {
  return Array.from({ length: n }, (_, i) => 1 / Math.pow(i + offset, power));
}

/**
 * MC Basic: for each (s,a), run numEpisodes episodes and average returns.
 * This is the textbook model-free conversion of policy iteration.
 */
export function mcBasic(
  config: GridWorldConfig,
  numEpisodesPerPair: number = 20,
  maxSteps: number = 30
): { qValues: number[][]; returns: number[][][] } {
  const numStates = config.rows * config.cols;
  const numActions = 5;
  const returns: number[][][] = Array.from({ length: numStates }, () =>
    Array.from({ length: numActions }, () => [])
  );

  // Use a uniform random behavior policy for generating episodes
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

  return { qValues, returns };
}

/**
 * n-step Sarsa: on-policy TD control with n-step returns.
 * Returns history of Q-values (one entry per episode).
 */
export function nStepSarsa(
  config: GridWorldConfig,
  alpha: number = 0.1,
  epsilon: number = 0.3,
  n: number = 3,
  episodes: number = 200,
  maxSteps: number = 30
): number[][][] {
  const numStates = config.rows * config.cols;
  const numActions = 5;
  let q = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  const history: number[][][] = [q.map((row) => [...row])];

  for (let ep = 0; ep < episodes; ep++) {
    let state = config.startState;
    const states: number[] = [state];
    const actions: Action[] = [];
    const rewards: number[] = [0];

    let policy = epsilonGreedyPolicy(q, epsilon);
    let action = sampleAction(policy[state]);
    actions.push(action);

    let t = 0;
    let T = Infinity;

    while (true) {
      if (t < T) {
        if (isTerminal(state, config)) {
          T = t;
        } else {
          const sNext = nextState(state, action, config);
          const r = reward(sNext, config);
          states.push(sNext);
          rewards.push(r);
          policy = epsilonGreedyPolicy(q, epsilon);
          const aNext = sampleAction(policy[sNext]);
          actions.push(aNext);
        }
      }

      const tau = t - n + 1;
      if (tau >= 0) {
        let G = 0;
        for (let i = tau + 1; i <= Math.min(tau + n, T); i++) {
          G += Math.pow(config.gamma, i - tau - 1) * rewards[i];
        }
        if (tau + n < T) {
          G += Math.pow(config.gamma, n) * q[states[tau + n]][actions[tau + n]];
        }
        const sTau = states[tau];
        const aTau = actions[tau];
        q[sTau][aTau] += alpha * (G - q[sTau][aTau]);
      }

      if (tau === T - 1) break;

      state = states[t + 1];
      action = actions[t + 1];
      t++;
      if (t > maxSteps) break;
    }

    history.push(q.map((row) => [...row]));
  }

  return history;
}

/**
 * REINFORCE with baseline for a multi-armed bandit.
 * baseline is updated incrementally.
 */
export function reinforceWithBaseline(
  actionRewards: number[],
  initialTheta: number[],
  alpha: number,
  beta: number,
  episodes: number
): {
  thetaHistory: number[][];
  policyHistory: number[][];
  rewardHistory: number[];
  baselineHistory: number[];
} {
  const theta = [...initialTheta];
  let baseline = 0;
  const thetaHistory: number[][] = [[...theta]];
  const policyHistory: number[][] = [softmaxPolicy(theta)];
  const rewardHistory: number[] = [];
  const baselineHistory: number[] = [baseline];

  for (let ep = 0; ep < episodes; ep++) {
    const policy = softmaxPolicy(theta);
    const action = sampleAction(policy);
    const r = actionRewards[action] + (Math.random() * 2 - 1) * 0.5;
    rewardHistory.push(r);

    baseline += beta * (r - baseline);
    baselineHistory.push(baseline);

    const advantage = r - baseline;
    for (let a = 0; a < theta.length; a++) {
      const indicator = a === action ? 1 : 0;
      theta[a] += alpha * advantage * (indicator - policy[a]);
    }

    thetaHistory.push([...theta]);
    policyHistory.push(softmaxPolicy(theta));
  }

  return { thetaHistory, policyHistory, rewardHistory, baselineHistory };
}
