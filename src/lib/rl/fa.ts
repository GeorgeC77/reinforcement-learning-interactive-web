/**
 * Function approximation utilities for reinforcement learning demos.
 *
 * Includes linear feature construction, semi-gradient TD(lambda), LSTD, and a tiny
 * hand-written MLP for a DQN-style tabular/continuous-state demo.
 */

import { mulberry32 } from './stochasticApproximation';
import {
  type GridWorldConfig,
  type Policy,
  type StateValues,
  type Action,
  step,
  isTerminal,
  sampleActionWithRng,
  solveStateValues,
  policyBellmanResidualV,
  policyBellmanResidualQ,
  optimalBellmanResidualQ,
  epsilonGreedyPolicy,
  greedyPolicy,
  estimateTrueActionValues,
  qTableRMSE,
} from './gridworld';

// ---------------------------------------------------------------------------
// Feature constructors
// ---------------------------------------------------------------------------

export function oneHotFeatures(state: number, config: GridWorldConfig): number[] {
  const vec = new Array(config.rows * config.cols).fill(0);
  vec[state] = 1;
  return vec;
}

export function coordinateFeatures(state: number, config: GridWorldConfig): number[] {
  const row = Math.floor(state / config.cols);
  const col = state % config.cols;
  const rNorm = config.rows > 1 ? (row / (config.rows - 1)) * 2 - 1 : 0;
  const cNorm = config.cols > 1 ? (col / (config.cols - 1)) * 2 - 1 : 0;
  return [1, rNorm, cNorm];
}

export function polynomialCoordinateFeatures(
  state: number,
  config: GridWorldConfig,
  degree: number
): number[] {
  const row = Math.floor(state / config.cols);
  const col = state % config.cols;
  const rNorm = config.rows > 1 ? (row / (config.rows - 1)) * 2 - 1 : 0;
  const cNorm = config.cols > 1 ? (col / (config.cols - 1)) * 2 - 1 : 0;
  const phi: number[] = [1];
  for (let d = 1; d <= degree; d++) {
    for (let i = 0; i <= d; i++) {
      phi.push(Math.pow(rNorm, i) * Math.pow(cNorm, d - i));
    }
  }
  return phi;
}

export type FeatureMode = 'onehot' | 'coordinate' | 'polynomial' | 'distance';

export function distanceStateFeatures(state: number, config: GridWorldConfig): number[] {
  const row = Math.floor(state / config.cols);
  const col = state % config.cols;
  const rNorm = config.rows > 1 ? (row / (config.rows - 1)) * 2 - 1 : 0;
  const cNorm = config.cols > 1 ? (col / (config.cols - 1)) * 2 - 1 : 0;
  const tRow = Math.floor(config.targetState / config.cols);
  const tCol = config.targetState % config.cols;
  const distanceToTarget = Math.sqrt((row - tRow) ** 2 + (col - tCol) ** 2);
  const maxDist = Math.sqrt((config.rows - 1) ** 2 + (config.cols - 1) ** 2);
  const isForbidden = config.forbiddenStates.includes(state) ? 1 : 0;
  return [1, rNorm, cNorm, distanceToTarget / Math.max(1, maxDist), isForbidden];
}

function makeFeatureFn(mode: FeatureMode, degree?: number) {
  return (state: number, config: GridWorldConfig) => {
    if (mode === 'onehot') return oneHotFeatures(state, config);
    if (mode === 'coordinate') return coordinateFeatures(state, config);
    if (mode === 'distance') return distanceStateFeatures(state, config);
    return polynomialCoordinateFeatures(state, config, degree ?? 2);
  };
}

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, x, i) => sum + x * (b[i] ?? 0), 0);
}

// ---------------------------------------------------------------------------
// Stationary distribution under a policy
// ---------------------------------------------------------------------------

export function stationaryDistribution(
  policy: Policy,
  config: GridWorldConfig,
  maxIter = 2000
): number[] {
  const numStates = config.rows * config.cols;
  let d = new Array(numStates).fill(1 / numStates);

  for (let it = 0; it < maxIter; it++) {
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
    const maxDiff = Math.max(...norm.map((v, i) => Math.abs(v - d[i])));
    d = norm;
    if (maxDiff < 1e-12) break;
  }

  return d;
}

// ---------------------------------------------------------------------------
// Semi-gradient TD(lambda) for state-value prediction
// ---------------------------------------------------------------------------

export interface SemiGradientTDOptions {
  alpha: number;
  lambda: number;
  featureMode: FeatureMode;
  polynomialDegree?: number;
  episodes: number;
  maxSteps: number;
  seed?: number;
}

export interface SemiGradientTDResult {
  valuesHistory: StateValues[];
  weightsHistory: number[][];
  trueValues: StateValues;
  visitCounts: number[];
  visitCountsHistory: number[][];
  residualHistory: number[];
}

export function semiGradientTD(
  policy: Policy,
  config: GridWorldConfig,
  options: SemiGradientTDOptions
): SemiGradientTDResult {
  const { alpha, lambda, featureMode, polynomialDegree, episodes, maxSteps, seed = 1 } = options;
  const rng = mulberry32(seed);
  const phi = makeFeatureFn(featureMode, polynomialDegree);
  const numStates = config.rows * config.cols;

  const trueValues = solveStateValues(policy, config);
  const featureDim = phi(0, config).length;
  let w = new Array(featureDim).fill(0);

  const valuesHistory: StateValues[] = [];
  const weightsHistory: number[][] = [];
  const visitCounts = new Array(numStates).fill(0);
  const visitCountsHistory: number[][] = [];
  const residualHistory: number[] = [];

  for (let ep = 0; ep < episodes; ep++) {
    let state = config.startState;
    let z = new Array(featureDim).fill(0);

    for (let stepIdx = 0; stepIdx < maxSteps; stepIdx++) {
      if (isTerminal(state, config)) break;
      visitCounts[state]++;
      const action = sampleActionWithRng(policy[state], rng);
      const result = step(state, action, config);

      const phiS = phi(state, config);
      const phiNext = phi(result.nextState, config);
      const vHat = dot(phiS, w);
      const vNext = result.done ? 0 : dot(phiNext, w);
      const delta = result.reward + config.gamma * vNext - vHat;

      for (let i = 0; i < featureDim; i++) {
        z[i] = config.gamma * lambda * z[i] + phiS[i];
      }

      for (let i = 0; i < featureDim; i++) {
        w[i] += alpha * delta * z[i];
      }

      state = result.nextState;
      if (result.done) break;
    }

    const predicted = Array.from({ length: numStates }, (_, s) => dot(phi(s, config), w));
    valuesHistory.push(predicted);
    weightsHistory.push([...w]);
    visitCountsHistory.push([...visitCounts]);
    residualHistory.push(policyBellmanResidualV(predicted, policy, config));
  }

  return { valuesHistory, weightsHistory, trueValues, visitCounts, visitCountsHistory, residualHistory };
}

// ---------------------------------------------------------------------------
// Tiny MLP for DQN demo
// ---------------------------------------------------------------------------

export interface MLPGradients {
  dW1: number[][];
  db1: number[];
  dW2: number[][];
  db2: number[];
}

export class SimpleMLP {
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
  W1: number[][];
  b1: number[];
  W2: number[][];
  b2: number[];

  constructor(inputSize: number, hiddenSize: number, outputSize: number, rng?: () => number) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.W1 = randomMatrix(inputSize, hiddenSize, 0.1, rng);
    this.b1 = new Array(hiddenSize).fill(0);
    this.W2 = randomMatrix(hiddenSize, outputSize, 0.1, rng);
    this.b2 = new Array(outputSize).fill(0);
  }

  forward(x: number[]): { out: number[]; hidden: number[] } {
    const hidden = new Array(this.hiddenSize).fill(0);
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < this.inputSize; i++) {
        sum += x[i] * this.W1[i][j];
      }
      hidden[j] = Math.max(0, sum);
    }

    const out = new Array(this.outputSize).fill(0);
    for (let k = 0; k < this.outputSize; k++) {
      let sum = this.b2[k];
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += hidden[j] * this.W2[j][k];
      }
      out[k] = sum;
    }
    return { out, hidden };
  }

  computeGradients(x: number[], action: number, target: number): { loss: number; grads: MLPGradients } {
    const { out, hidden } = this.forward(x);
    const error = target - out[action];
    const loss = 0.5 * error * error;

    const dOut = new Array(this.outputSize).fill(0);
    dOut[action] = -error;

    const dW2: number[][] = Array.from({ length: this.hiddenSize }, () => new Array(this.outputSize).fill(0));
    const db2 = [...dOut];
    for (let j = 0; j < this.hiddenSize; j++) {
      for (let k = 0; k < this.outputSize; k++) {
        dW2[j][k] = hidden[j] * dOut[k];
      }
    }

    const dHidden = new Array(this.hiddenSize).fill(0);
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = 0;
      for (let k = 0; k < this.outputSize; k++) {
        sum += dOut[k] * this.W2[j][k];
      }
      dHidden[j] = hidden[j] > 0 ? sum : 0;
    }

    const dW1: number[][] = Array.from({ length: this.inputSize }, () => new Array(this.hiddenSize).fill(0));
    const db1 = [...dHidden];
    for (let i = 0; i < this.inputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        dW1[i][j] = x[i] * dHidden[j];
      }
    }

    return { loss, grads: { dW1, db1, dW2, db2 } };
  }

  applyGradients(grads: MLPGradients, alpha: number): void {
    for (let j = 0; j < this.hiddenSize; j++) {
      this.b1[j] -= alpha * grads.db1[j];
      for (let i = 0; i < this.inputSize; i++) {
        this.W1[i][j] -= alpha * grads.dW1[i][j];
      }
    }
    for (let k = 0; k < this.outputSize; k++) {
      this.b2[k] -= alpha * grads.db2[k];
      for (let j = 0; j < this.hiddenSize; j++) {
        this.W2[j][k] -= alpha * grads.dW2[j][k];
      }
    }
  }

  trainStep(x: number[], action: number, target: number, alpha: number): number {
    const { loss, grads } = this.computeGradients(x, action, target);
    this.applyGradients(grads, alpha);
    return loss;
  }

  copy(): SimpleMLP {
    const net = new SimpleMLP(this.inputSize, this.hiddenSize, this.outputSize, () => 0);
    net.W1 = this.W1.map((row) => [...row]);
    net.b1 = [...this.b1];
    net.W2 = this.W2.map((row) => [...row]);
    net.b2 = [...this.b2];
    return net;
  }

  initFrom(other: SimpleMLP) {
    this.W1 = other.W1.map((row) => [...row]);
    this.b1 = [...other.b1];
    this.W2 = other.W2.map((row) => [...row]);
    this.b2 = [...other.b2];
  }
}

function randomMatrix(
  rows: number,
  cols: number,
  scale: number,
  rng?: () => number
): number[][] {
  const rand = rng;
  if (!rand) {
    throw new Error('randomMatrix requires an rng function; Math.random is not allowed');
  }
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (rand() * 2 - 1) * scale)
  );
}

// ---------------------------------------------------------------------------
// DQN on the GridWorld using coordinate features
// ---------------------------------------------------------------------------

export type EpsilonScheduleMode = 'fixed' | 'decay-floor' | 'glie';

function epsilonForEpisode(
  episode: number,
  epsilon0: number,
  mode: EpsilonScheduleMode,
  epsilonMin = 0.01
): number {
  if (mode === 'fixed') return epsilon0;
  if (mode === 'glie') return epsilon0 / Math.sqrt(episode + 1);
  return Math.max(epsilonMin, epsilon0 * Math.pow(0.99, episode));
}

export interface DQNOptions {
  hiddenSize: number;
  alpha: number;
  epsilon: number;
  epsilonMode?: EpsilonScheduleMode;
  epsilonMin?: number;
  gamma?: number;
  batchSize: number;
  replayCapacity: number;
  targetUpdateInterval: number;
  episodes: number;
  maxSteps: number;
  seed?: number;
}

interface Transition {
  state: number;
  action: Action;
  reward: number;
  nextState: number;
  done: boolean;
}

export interface DQNBatchItem {
  state: number;
  action: Action;
  reward: number;
  nextState: number;
  done: boolean;
  target: number;
  prediction: number;
  loss: number;
}

export interface DQNUpdateRecord {
  update: number;
  environmentStep: number;
  episode: number;
  batchLoss: number;
  qTable: number[][];
  qRmse: number;
  optimalityResidual: number;
  replaySize: number;
  targetSynced: boolean;
}

export interface DQNEpisodeRecord {
  episode: number;
  cumulativeReward: number;
  episodeLength: number;
  success: boolean;
  truncated: boolean;
  qTableAfterEpisode: number[][];
}

export interface DQNResult {
  updateHistory: DQNUpdateRecord[];
  episodeHistory: DQNEpisodeRecord[];
  finalReplaySize: number;
  lastBatch: DQNBatchItem[];
  visitCounts: number[];
}

export function dqnGridWorld(config: GridWorldConfig, options: DQNOptions): DQNResult {
  const {
    hiddenSize,
    alpha,
    epsilon,
    epsilonMode = 'fixed',
    epsilonMin = 0.01,
    gamma = config.gamma,
    batchSize,
    replayCapacity,
    targetUpdateInterval,
    episodes,
    maxSteps,
    seed = 1,
  } = options;
  const rng = mulberry32(seed);

  const numStates = config.rows * config.cols;
  const numActions = 5;
  const featureDim = 3; // [1, rNorm, cNorm]

  const mainNet = new SimpleMLP(featureDim, hiddenSize, numActions, rng);
  const targetNet = mainNet.copy();
  const replay: Transition[] = [];

  const qStar = estimateTrueActionValues(config);

  const updateHistory: DQNUpdateRecord[] = [];
  const episodeHistory: DQNEpisodeRecord[] = [];
  const visitCounts = new Array(numStates).fill(0);
  let lastBatch: DQNBatchItem[] = [];
  let trainUpdateCount = 0;
  let environmentStepCount = 0;

  function stateFeatures(state: number): number[] {
    return coordinateFeatures(state, config);
  }

  function networkQ(net: SimpleMLP, state: number): number[] {
    return net.forward(stateFeatures(state)).out;
  }

  function selectAction(state: number, eps: number): Action {
    const qValues = networkQ(mainNet, state);
    if (rng() < eps) {
      return Math.floor(rng() * numActions) as Action;
    }
    const maxQ = Math.max(...qValues);
    const best = qValues
      .map((q, i) => ({ q, i }))
      .filter(({ q }) => Math.abs(q - maxQ) < 1e-6)
      .map(({ i }) => i);
    return best[Math.floor(rng() * best.length)] as Action;
  }

  function buildQTable(): number[][] {
    const table: number[][] = [];
    for (let s = 0; s < numStates; s++) {
      table.push(networkQ(mainNet, s));
    }
    return table;
  }

  function zeroGradients(): MLPGradients {
    return {
      dW1: Array.from({ length: featureDim }, () => new Array(hiddenSize).fill(0)),
      db1: new Array(hiddenSize).fill(0),
      dW2: Array.from({ length: hiddenSize }, () => new Array(numActions).fill(0)),
      db2: new Array(numActions).fill(0),
    };
  }

  function addGradients(a: MLPGradients, b: MLPGradients): MLPGradients {
    return {
      dW1: a.dW1.map((row, i) => row.map((v, j) => v + b.dW1[i][j])),
      db1: a.db1.map((v, i) => v + b.db1[i]),
      dW2: a.dW2.map((row, i) => row.map((v, j) => v + b.dW2[i][j])),
      db2: a.db2.map((v, i) => v + b.db2[i]),
    };
  }

  function scaleGradients(g: MLPGradients, s: number): MLPGradients {
    return {
      dW1: g.dW1.map((row) => row.map((v) => v * s)),
      db1: g.db1.map((v) => v * s),
      dW2: g.dW2.map((row) => row.map((v) => v * s)),
      db2: g.db2.map((v) => v * s),
    };
  }

  for (let ep = 0; ep < episodes; ep++) {
    const eps = epsilonForEpisode(ep, epsilon, epsilonMode, epsilonMin);
    let state = config.startState;
    let episodeReturn = 0;
    let episodeLength = 0;
    let reachedTarget = false;

    for (let stepIdx = 0; stepIdx < maxSteps; stepIdx++) {
      if (isTerminal(state, config)) break;
      visitCounts[state]++;
      episodeLength++;
      environmentStepCount++;

      const action = selectAction(state, eps);
      const result = step(state, action, config);
      episodeReturn += result.reward;

      replay.push({ state, action, reward: result.reward, nextState: result.nextState, done: result.done });
      if (replay.length > replayCapacity) replay.shift();

      if (replay.length >= batchSize) {
        const batch = sampleReplay(replay, batchSize, rng);
        let totalLoss = 0;
        const batchDetails: DQNBatchItem[] = [];
        let gradSum = zeroGradients();

        for (const trans of batch) {
          const qNext = networkQ(targetNet, trans.nextState);
          const maxQNext = Math.max(...qNext);
          const target = trans.done ? trans.reward : trans.reward + gamma * maxQNext;
          const prediction = networkQ(mainNet, trans.state)[trans.action];
          const { loss, grads } = mainNet.computeGradients(stateFeatures(trans.state), trans.action, target);
          gradSum = addGradients(gradSum, grads);
          totalLoss += loss;
          batchDetails.push({ ...trans, target, prediction, loss });
        }

        const avgGrad = scaleGradients(gradSum, 1 / batch.length);
        mainNet.applyGradients(avgGrad, alpha);

        lastBatch = batchDetails;
        trainUpdateCount++;
        let targetSynced = false;
        if (targetUpdateInterval > 0 && trainUpdateCount % targetUpdateInterval === 0) {
          targetNet.initFrom(mainNet);
          targetSynced = true;
        }

        const qTableNow = buildQTable();
        updateHistory.push({
          update: trainUpdateCount,
          environmentStep: environmentStepCount,
          episode: ep,
          batchLoss: totalLoss / batch.length,
          qTable: qTableNow,
          qRmse: qTableRMSE(qTableNow, qStar),
          optimalityResidual: optimalBellmanResidualQ(qTableNow, config),
          replaySize: replay.length,
          targetSynced,
        });
      }

      state = result.nextState;
      if (result.done) {
        if (state === config.targetState) reachedTarget = true;
        break;
      }
    }

    episodeHistory.push({
      episode: ep,
      cumulativeReward: episodeReturn,
      episodeLength,
      success: reachedTarget,
      truncated: !reachedTarget && episodeLength >= maxSteps,
      qTableAfterEpisode: buildQTable(),
    });
  }

  return {
    updateHistory,
    episodeHistory,
    finalReplaySize: replay.length,
    lastBatch,
    visitCounts,
  };
}

function sampleReplay<T>(buffer: T[], n: number, rng: () => number): T[] {
  const sample: T[] = [];
  const copy = [...buffer];
  for (let i = 0; i < n && copy.length > 0; i++) {
    const idx = Math.floor(rng() * copy.length);
    sample.push(copy[idx]);
    copy.splice(idx, 1);
  }
  return sample;
}

// ---------------------------------------------------------------------------
// Action-value function approximation
// ---------------------------------------------------------------------------

export type ActionValueFeatureMode = 'onehot' | 'shared';

export function actionValueFeatureDim(config: GridWorldConfig, mode: ActionValueFeatureMode): number {
  if (mode === 'onehot') return config.rows * config.cols * 5;
  const stateDim = distanceStateFeatures(0, config).length;
  const actionDim = 5;
  return stateDim + actionDim + stateDim * actionDim;
}

export function actionValueFeatures(
  state: number,
  action: Action,
  config: GridWorldConfig,
  mode: ActionValueFeatureMode
): number[] {
  if (mode === 'onehot') {
    const dim = config.rows * config.cols * 5;
    const vec = new Array(dim).fill(0);
    vec[state * 5 + action] = 1;
    return vec;
  }

  const stateFeat = distanceStateFeatures(state, config);
  const actionFeat = new Array(5).fill(0);
  actionFeat[action] = 1;
  const interaction: number[] = [];
  for (const sf of stateFeat) {
    for (const af of actionFeat) {
      interaction.push(sf * af);
    }
  }
  return [...stateFeat, ...actionFeat, ...interaction];
}

export interface ActionValueFAOptions {
  alpha: number;
  epsilon: number;
  epsilonMode?: EpsilonScheduleMode;
  epsilonMin?: number;
  gamma?: number;
  episodes: number;
  maxSteps: number;
  featureMode: ActionValueFeatureMode;
  algorithm: 'sarsa' | 'qlearning';
  seed?: number;
}

export interface LastFAUpdate {
  state: number;
  action: Action;
  reward: number;
  nextState: number;
  nextAction?: Action;
  prediction: number;
  target: number;
  tdError: number;
  weightChange: number;
}

export interface ActionValueFAResult {
  qHistory: number[][][];
  weightsHistory: number[][];
  lastUpdate: LastFAUpdate | null;
  residualHistory: number[];
  behaviorPolicyHistory: Policy[];
  greedyPolicyHistory: Policy[];
  episodeReturnHistory: number[];
  episodeLengthHistory: number[];
  visitCounts: number[];
}

export function actionValueFA(
  config: GridWorldConfig,
  options: ActionValueFAOptions
): ActionValueFAResult {
  const {
    alpha,
    epsilon,
    epsilonMode = 'fixed',
    epsilonMin = 0.01,
    gamma = config.gamma,
    episodes,
    maxSteps,
    featureMode,
    algorithm,
    seed = 1,
  } = options;
  const rng = mulberry32(seed);
  const numStates = config.rows * config.cols;
  const numActions = 5;
  const featureDim = actionValueFeatureDim(config, featureMode);
  let w = new Array(featureDim).fill(0);

  const qHistory: number[][][] = [];
  const weightsHistory: number[][] = [];
  const residualHistory: number[] = [];
  const behaviorPolicyHistory: Policy[] = [];
  const greedyPolicyHistory: Policy[] = [];
  const episodeReturnHistory: number[] = [];
  const episodeLengthHistory: number[] = [];
  const visitCounts = new Array(numStates).fill(0);
  let lastUpdate: LastFAUpdate | null = null;

  function qValue(state: number, action: Action): number {
    return dot(actionValueFeatures(state, action, config, featureMode), w);
  }

  function greedyAction(state: number): Action {
    const qs = Array.from({ length: numActions }, (_, a) => qValue(state, a as Action));
    const maxQ = Math.max(...qs);
    const best = qs
      .map((q, i) => ({ q, i }))
      .filter(({ q }) => Math.abs(q - maxQ) < 1e-6)
      .map(({ i }) => i);
    return best[Math.floor(rng() * best.length)] as Action;
  }

  function sampleEpsilonGreedy(state: number, eps: number): Action {
    if (rng() < eps) {
      return Math.floor(rng() * numActions) as Action;
    }
    return greedyAction(state);
  }

  for (let ep = 0; ep < episodes; ep++) {
    const eps = epsilonForEpisode(ep, epsilon, epsilonMode, epsilonMin);
    let state = config.startState;
    let action = sampleEpsilonGreedy(state, eps);
    let episodeReturn = 0;
    let episodeLength = 0;

    for (let stepIdx = 0; stepIdx < maxSteps; stepIdx++) {
      if (isTerminal(state, config)) break;
      visitCounts[state]++;
      episodeLength++;
      const result = step(state, action, config);
      episodeReturn += result.reward;
      const phi = actionValueFeatures(state, action, config, featureMode);
      const prediction = dot(phi, w);

      let target = result.reward;
      let nextAction: Action | undefined;
      if (!result.done) {
        if (algorithm === 'sarsa') {
          nextAction = sampleEpsilonGreedy(result.nextState, eps);
          target += gamma * qValue(result.nextState, nextAction);
        } else {
          const maxQ = Math.max(
            ...Array.from({ length: numActions }, (_, a) => qValue(result.nextState, a as Action))
          );
          target += gamma * maxQ;
        }
      }

      const tdError = target - prediction;
      const weightBefore = w.map((x) => x);
      for (let i = 0; i < featureDim; i++) {
        w[i] += alpha * tdError * phi[i];
      }
      const weightChange = Math.sqrt(w.reduce((sum, wi, i) => sum + (wi - weightBefore[i]) ** 2, 0));

      lastUpdate = {
        state,
        action,
        reward: result.reward,
        nextState: result.nextState,
        nextAction,
        prediction,
        target,
        tdError,
        weightChange,
      };

      state = result.nextState;
      action = algorithm === 'sarsa' && nextAction !== undefined ? nextAction : sampleEpsilonGreedy(state, eps);
      if (result.done) break;
    }

    episodeReturnHistory.push(episodeReturn);
    episodeLengthHistory.push(episodeLength);

    const qTable: number[][] = [];
    for (let s = 0; s < numStates; s++) {
      const row: number[] = [];
      for (let a = 0; a < numActions; a++) {
        row.push(qValue(s, a as Action));
      }
      qTable.push(row);
    }
    qHistory.push(qTable);
    weightsHistory.push([...w]);

    const behaviorPolicy = epsilonGreedyPolicy(qTable, eps, rng);
    const greedyPol = greedyPolicy(qTable);
    behaviorPolicyHistory.push(behaviorPolicy);
    greedyPolicyHistory.push(greedyPol);

    residualHistory.push(
      algorithm === 'qlearning'
        ? optimalBellmanResidualQ(qTable, config)
        : policyBellmanResidualQ(qTable, behaviorPolicy, config)
    );
  }

  return {
    qHistory,
    weightsHistory,
    lastUpdate,
    residualHistory,
    behaviorPolicyHistory,
    greedyPolicyHistory,
    episodeReturnHistory,
    episodeLengthHistory,
    visitCounts,
  };
}

// ---------------------------------------------------------------------------
// LSTD and small matrix helpers
// ---------------------------------------------------------------------------

function matZero(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

function matAdd(a: number[][], b: number[][]): number[][] {
  return a.map((row, i) => row.map((v, j) => v + b[i][j]));
}

function outer(a: number[], b: number[]): number[][] {
  return a.map((x) => b.map((y) => x * y));
}

function matVec(a: number[][], v: number[]): number[] {
  return a.map((row) => row.reduce((sum, x, i) => sum + x * v[i], 0));
}

interface MatrixInvertResult {
  inverse: number[][];
  minPivot: number;
  conditionEstimate: number;
}

function invertMatrixWithStats(a: number[][]): MatrixInvertResult | null {
  const n = a.length;
  if (n === 0) return null;
  // augment [a | I]
  const aug = a.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (j === i ? 1 : 0))]);

  let minPivot = Infinity;
  let maxPivot = 0;

  for (let i = 0; i < n; i++) {
    let pivot = aug[i][i];
    let pivotRow = i;
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(aug[r][i]) > Math.abs(pivot)) {
        pivot = aug[r][i];
        pivotRow = r;
      }
    }
    const absPivot = Math.abs(pivot);
    if (absPivot < 1e-12) return null;
    minPivot = Math.min(minPivot, absPivot);
    maxPivot = Math.max(maxPivot, absPivot);
    if (pivotRow !== i) {
      [aug[i], aug[pivotRow]] = [aug[pivotRow], aug[i]];
    }
    for (let j = 0; j < 2 * n; j++) {
      aug[i][j] /= pivot;
    }
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const factor = aug[r][i];
      for (let j = 0; j < 2 * n; j++) {
        aug[r][j] -= factor * aug[i][j];
      }
    }
  }

  return {
    inverse: aug.map((row) => row.slice(n)),
    minPivot,
    conditionEstimate: minPivot === 0 ? Infinity : maxPivot / minPivot,
  };
}

export type LSTDResult =
  | {
      ok: true;
      A: number[][];
      b: number[];
      w: number[];
      minPivot: number;
      conditionEstimate: number;
      ridgeLambda: number;
    }
  | {
      ok: false;
      reason: 'singular' | 'near-singular' | 'insufficient-coverage';
      A: number[][];
      b: number[];
      minPivot: number;
      conditionEstimate: number;
    };

const LSTD_RIDGE_CANDIDATES = [0, 1e-8, 1e-6, 1e-4, 1e-2];

export function lstdFromTrajectory(
  policy: Policy,
  config: GridWorldConfig,
  options: {
    featureMode: FeatureMode;
    polynomialDegree?: number;
    episodes?: number;
    maxSteps?: number;
    seed?: number;
    ridgeLambda?: number;
  }
): LSTDResult {
  const { featureMode, polynomialDegree, episodes = 200, maxSteps = 50, seed = 1, ridgeLambda } = options;
  const rng = mulberry32(seed);
  const phi = makeFeatureFn(featureMode, polynomialDegree);
  const featureDim = phi(0, config).length;
  const gamma = config.gamma;

  let A = matZero(featureDim, featureDim);
  let bVec = new Array(featureDim).fill(0);
  let transitionCount = 0;

  for (let ep = 0; ep < episodes; ep++) {
    let state = config.startState;
    for (let stepIdx = 0; stepIdx < maxSteps; stepIdx++) {
      if (isTerminal(state, config)) break;
      const action = sampleActionWithRng(policy[state], rng);
      const result = step(state, action, config);
      const phiS = phi(state, config);
      const phiNext = result.done ? new Array(featureDim).fill(0) : phi(result.nextState, config);
      const diff = phiS.map((v, i) => v - gamma * phiNext[i]);
      A = matAdd(A, outer(phiS, diff));
      bVec = bVec.map((v, i) => v + result.reward * phiS[i]);
      transitionCount++;
      state = result.nextState;
      if (result.done) break;
    }
  }

  const maxDiagonal = Math.max(...A.map((row, i) => Math.abs(row[i])));
  if (transitionCount < featureDim || maxDiagonal < 1e-12) {
    return {
      ok: false,
      reason: 'insufficient-coverage',
      A,
      b: bVec,
      minPivot: 0,
      conditionEstimate: Infinity,
    };
  }

  const ridgeValues = ridgeLambda !== undefined ? [ridgeLambda] : LSTD_RIDGE_CANDIDATES;

  for (const ridge of ridgeValues) {
    const Areg = A.map((row, i) => row.map((v, j) => (i === j ? v + ridge : v)));
    const invStats = invertMatrixWithStats(Areg);
    if (invStats) {
      return {
        ok: true,
        A,
        b: bVec,
        w: matVec(invStats.inverse, bVec),
        minPivot: invStats.minPivot,
        conditionEstimate: invStats.conditionEstimate,
        ridgeLambda: ridge,
      };
    }
  }

  // All candidates failed: report stats from the unregularized matrix.
  const unregStats = invertMatrixWithStats(A);
  return {
    ok: false,
    reason: 'near-singular',
    A,
    b: bVec,
    minPivot: unregStats?.minPivot ?? 0,
    conditionEstimate: unregStats?.conditionEstimate ?? Infinity,
  };
}


// ---------------------------------------------------------------------------
// Recursive Least Squares (RLS) for policy evaluation
// ---------------------------------------------------------------------------

export interface RLSTDOptions {
  featureMode: FeatureMode;
  polynomialDegree?: number;
  episodes?: number;
  maxSteps?: number;
  seed?: number;
  /** Initial diagonal precision δ; P_0 = (1/δ) I. */
  delta?: number;
}

export interface RLSTDResult {
  weightsHistory: number[][];
  valuesHistory: StateValues[];
  trueValues: StateValues;
  stepsProcessed: number;
}

function matScale(a: number[][], s: number): number[][] {
  return a.map((row) => row.map((v) => v * s));
}

/**
 * Recursive least squares update of the LSTD estimate w = A^{-1} b with
 * A = Σ φ(φ − γφ')ᵀ and b = Σ r φ, using the Sherman–Morrison identity
 * after every transition. Returns the per-episode weight/value history.
 */
export function rlsTD(
  policy: Policy,
  config: GridWorldConfig,
  options: RLSTDOptions
): RLSTDResult {
  const {
    featureMode,
    polynomialDegree,
    episodes = 200,
    maxSteps = 50,
    seed = 1,
    delta = 1e-3,
  } = options;
  const rng = mulberry32(seed);
  const phi = makeFeatureFn(featureMode, polynomialDegree);
  const featureDim = phi(0, config).length;
  const numStates = config.rows * config.cols;
  const gamma = config.gamma;

  const trueValues = solveStateValues(policy, config);

  // P = (1/δ) I, w = 0, b = 0.
  const P = matZero(featureDim, featureDim).map((row, i) =>
    row.map((_, j) => (i === j ? 1 / delta : 0))
  );
  const bVec = new Array(featureDim).fill(0);

  const weightsHistory: number[][] = [];
  const valuesHistory: StateValues[] = [];
  let stepsProcessed = 0;

  for (let ep = 0; ep < episodes; ep++) {
    let state = config.startState;
    for (let stepIdx = 0; stepIdx < maxSteps; stepIdx++) {
      if (isTerminal(state, config)) break;
      const action = sampleActionWithRng(policy[state], rng);
      const result = step(state, action, config);
      const phiS = phi(state, config);
      const phiNext = result.done ? new Array(featureDim).fill(0) : phi(result.nextState, config);
      const d = phiS.map((v, i) => v - gamma * phiNext[i]);

      // Rank-1 Sherman–Morrison update of P = A^{-1} for A ← A + φ dᵀ:
      // P ← P − (P φ)(dᵀ P) / (1 + dᵀ P φ)
      const Pphi = matVec(P, phiS);
      const dTP = new Array(featureDim).fill(0);
      for (let j = 0; j < featureDim; j++) {
        for (let i = 0; i < featureDim; i++) dTP[j] += d[i] * P[i][j];
      }
      const denom = 1 + d.reduce((s, v, i) => s + v * Pphi[i], 0);
      if (denom > 1e-12) {
        const corr = matScale(outer(Pphi, dTP), 1 / denom);
        for (let i = 0; i < featureDim; i++) {
          for (let j = 0; j < featureDim; j++) {
            P[i][j] -= corr[i][j];
          }
        }
      }
      for (let i = 0; i < featureDim; i++) bVec[i] += result.reward * phiS[i];

      state = result.nextState;
      stepsProcessed++;
      if (result.done) break;
    }

    const w = matVec(P, bVec);
    weightsHistory.push([...w]);
    valuesHistory.push(Array.from({ length: numStates }, (_, s) => dot(phi(s, config), w)));
  }

  return { weightsHistory, valuesHistory, trueValues, stepsProcessed };
}
