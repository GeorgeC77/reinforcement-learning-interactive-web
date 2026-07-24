/**
 * Dyna-Q: integrate model-free Q-learning with a learned model and planning.
 *
 * After every real transition the agent updates its internal model
 * (transition/reward statistics per (s,a)), then performs a configurable
 * number of planning steps on simulated transitions drawn from that model.
 * planningSteps = 0 recovers plain Q-learning. All randomness is driven by a
 * caller-supplied seeded rng.
 */

import { mulberry32 } from './stochasticApproximation';
import {
  type GridWorldConfig,
  type Action,
  step,
  isTerminal,
  estimateTrueActionValues,
  qTableRMSE,
} from './gridworld';

export interface DynaQOptions {
  episodes: number;
  maxSteps: number;
  alpha: number;
  epsilon: number;
  /** Number of planning updates per real environment step. */
  planningSteps: number;
  seed?: number;
}

export interface DynaEpisode {
  episode: number;
  steps: number;
  cumulativeReward: number;
  /** RMSE of the learned q table against the DP reference q*. */
  qRmse: number;
  /** Fraction of visited (s,a) whose model argmax transition matches the environment. */
  modelAccuracy: number;
  q: number[][];
}

export interface DynaQResult {
  episodes: DynaEpisode[];
  finalQ: number[][];
  /** Learned most-likely next state for each (s,a), -1 if never visited. */
  modelNextState: number[][];
  /** Learned mean reward for each (s,a), 0 if never visited. */
  modelReward: number[][];
  visitCounts: number[][];
  totalEnvSteps: number;
  totalPlanningSteps: number;
}

function argmax(values: number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

export function dynaQ(config: GridWorldConfig, options: DynaQOptions): DynaQResult {
  const { episodes, maxSteps, alpha, epsilon, planningSteps, seed = 1 } = options;
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;
  const gamma = config.gamma;
  const qStar = estimateTrueActionValues(config);

  const q = Array.from({ length: numStates }, () => new Array(numActions).fill(0));

  // Model statistics per (s,a).
  const counts = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  const rewardSum = Array.from({ length: numStates }, () => new Array(numActions).fill(0));
  // Deterministic-model assumption: store the first observed (s', r) pair,
  // then keep a running mean of rewards. This is the classic Dyna-Q model
  // for deterministic environments.
  const modelNext = Array.from({ length: numStates }, () => new Array(numActions).fill(-1));
  const modelR = Array.from({ length: numStates }, () => new Array(numActions).fill(0));

  // Pool of previously visited (s,a) pairs for planning.
  const visited: { s: number; a: number }[] = [];

  const episodeRecords: DynaEpisode[] = [];
  let totalEnvSteps = 0;
  let totalPlanningSteps = 0;

  for (let ep = 0; ep < episodes; ep++) {
    let state = config.startState;
    let stepsThisEpisode = 0;
    let cumulativeReward = 0;

    for (let t = 0; t < maxSteps; t++) {
      if (isTerminal(state, config)) break;

      // ε-greedy action.
      let action: Action;
      if (rng() < epsilon) {
        action = Math.floor(rng() * numActions) as Action;
      } else {
        action = argmax(q[state]) as Action;
      }

      const result = step(state, action, config);
      const sNext = result.nextState;
      const r = result.reward;

      // Real Q-learning update.
      q[state][action] +=
        alpha * (r + gamma * (result.done ? 0 : Math.max(...q[sNext])) - q[state][action]);

      // Model update (first observation fixes the next state; reward averages).
      if (counts[state][action] === 0) {
        visited.push({ s: state, a: action });
        modelNext[state][action] = sNext;
      }
      counts[state][action]++;
      rewardSum[state][action] += r;
      modelR[state][action] = rewardSum[state][action] / counts[state][action];

      // Planning: replay simulated transitions from the model.
      for (let p = 0; p < planningSteps && visited.length > 0; p++) {
        const { s, a } = visited[Math.floor(rng() * visited.length)];
        const ns = modelNext[s][a];
        const mr = modelR[s][a];
        const done = isTerminal(ns, config);
        q[s][a] += alpha * (mr + gamma * (done ? 0 : Math.max(...q[ns])) - q[s][a]);
        totalPlanningSteps++;
      }

      state = sNext;
      stepsThisEpisode++;
      cumulativeReward += r;
      totalEnvSteps++;
      if (result.done) break;
    }

    // Model accuracy: fraction of visited (s,a) whose argmax model transition
    // matches the deterministic environment transition.
    let accurate = 0;
    for (const { s, a } of visited) {
      if (modelNext[s][a] === step(s, a as Action, config).nextState) accurate++;
    }

    episodeRecords.push({
      episode: ep,
      steps: stepsThisEpisode,
      cumulativeReward,
      qRmse: qTableRMSE(q, qStar),
      modelAccuracy: visited.length === 0 ? 0 : accurate / visited.length,
      q: q.map((row) => [...row]),
    });
  }

  return {
    episodes: episodeRecords,
    finalQ: q,
    modelNextState: modelNext,
    modelReward: modelR,
    visitCounts: counts,
    totalEnvSteps,
    totalPlanningSteps,
  };
}
