/**
 * Function approximation utilities for reinforcement learning demos.
 *
 * Includes linear feature construction, semi-gradient TD(lambda), and a tiny
 * hand-written MLP for a DQN-style tabular/continuous-state demo.
 */

import {
  type GridWorldConfig,
  type Policy,
  type StateValues,
  type Action,
  step,
  isTerminal,
  sampleAction,
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

export type FeatureMode = 'onehot' | 'coordinate' | 'polynomial';

function makeFeatureFn(mode: FeatureMode, degree?: number) {
  return (state: number, config: GridWorldConfig) => {
    if (mode === 'onehot') return oneHotFeatures(state, config);
    if (mode === 'coordinate') return coordinateFeatures(state, config);
    return polynomialCoordinateFeatures(state, config, degree ?? 2);
  };
}

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, x, i) => sum + x * (b[i] ?? 0), 0);
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
}

export function semiGradientTD(
  policy: Policy,
  config: GridWorldConfig,
  options: SemiGradientTDOptions
): {
  valuesHistory: StateValues[];
  weightsHistory: number[][];
  trueValues: StateValues;
} {
  const { alpha, lambda, featureMode, polynomialDegree, episodes, maxSteps } = options;
  const phi = makeFeatureFn(featureMode, polynomialDegree);
  const numStates = config.rows * config.cols;

  // Compute true state values under the policy for reference
  const trueValues = solveStateValues(policy, config);

  const featureDim = phi(0, config).length;
  let w = new Array(featureDim).fill(0);
  const valuesHistory: StateValues[] = [];
  const weightsHistory: number[][] = [];

  for (let ep = 0; ep < episodes; ep++) {
    let state = config.startState;
    let z = new Array(featureDim).fill(0);

    for (let stepIdx = 0; stepIdx < maxSteps; stepIdx++) {
      if (isTerminal(state, config)) break;
      const action = sampleAction(policy[state]);
      const result = step(state, action, config);

      const phiS = phi(state, config);
      const phiNext = phi(result.nextState, config);
      const vHat = dot(phiS, w);
      const vNext = result.done ? 0 : dot(phiNext, w);
      const delta = result.reward + config.gamma * vNext - vHat;

      // Accumulating eligibility traces
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
  }

  return { valuesHistory, weightsHistory, trueValues };
}

function solveStateValues(policy: Policy, config: GridWorldConfig): StateValues {
  const numStates = config.rows * config.cols;
  const rPi = new Array(numStates).fill(0);
  const pPi: number[][] = Array.from({ length: numStates }, () => new Array(numStates).fill(0));

  for (let s = 0; s < numStates; s++) {
    for (let a = 0; a < 5; a++) {
      const prob = policy[s][a];
      if (prob === 0) continue;
      const result = step(s, a as Action, config);
      rPi[s] += prob * result.reward;
      pPi[s][result.nextState] += prob;
    }
  }

  const A: number[][] = Array.from({ length: numStates }, (_, i) =>
    Array.from({ length: numStates }, (_, j) => (i === j ? 1 : 0) - config.gamma * pPi[i][j])
  );
  return solveLinearSystem(A, rPi);
}

function solveLinearSystem(A: number[][], b: number[]): number[] {
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

// ---------------------------------------------------------------------------
// Tiny MLP for DQN demo
// ---------------------------------------------------------------------------

export interface DQNOptions {
  hiddenSize: number;
  alpha: number;
  epsilon: number;
  gamma?: number;
  batchSize: number;
  replayCapacity: number;
  targetUpdateInterval: number;
  episodes: number;
  maxSteps: number;
}

interface Transition {
  state: number;
  action: Action;
  reward: number;
  nextState: number;
  done: boolean;
}

class SimpleMLP {
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
  W1: number[][];
  b1: number[];
  W2: number[][];
  b2: number[];

  constructor(inputSize: number, hiddenSize: number, outputSize: number) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.W1 = randomMatrix(inputSize, hiddenSize, 0.1);
    this.b1 = new Array(hiddenSize).fill(0);
    this.W2 = randomMatrix(hiddenSize, outputSize, 0.1);
    this.b2 = new Array(outputSize).fill(0);
  }

  forward(x: number[]): { out: number[]; hidden: number[] } {
    const hidden = new Array(this.hiddenSize).fill(0);
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < this.inputSize; i++) {
        sum += x[i] * this.W1[i][j];
      }
      hidden[j] = Math.max(0, sum); // ReLU
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

  trainStep(x: number[], action: number, target: number, alpha: number): number {
    const { out, hidden } = this.forward(x);
    const error = target - out[action];
    const loss = error * error;

    // Output layer gradients for the chosen action only
    const dOut = new Array(this.outputSize).fill(0);
    dOut[action] = -error;

    // Gradients w.r.t. W2, b2
    const dW2: number[][] = Array.from({ length: this.hiddenSize }, () => new Array(this.outputSize).fill(0));
    const db2 = [...dOut];
    for (let j = 0; j < this.hiddenSize; j++) {
      for (let k = 0; k < this.outputSize; k++) {
        dW2[j][k] = hidden[j] * dOut[k];
      }
    }

    // Backprop into hidden layer
    const dHidden = new Array(this.hiddenSize).fill(0);
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = 0;
      for (let k = 0; k < this.outputSize; k++) {
        sum += dOut[k] * this.W2[j][k];
      }
      // ReLU derivative
      dHidden[j] = hidden[j] > 0 ? sum : 0;
    }

    const dW1: number[][] = Array.from({ length: this.inputSize }, () => new Array(this.hiddenSize).fill(0));
    const db1 = [...dHidden];
    for (let i = 0; i < this.inputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        dW1[i][j] = x[i] * dHidden[j];
      }
    }

    // Apply gradients
    for (let j = 0; j < this.hiddenSize; j++) {
      this.b1[j] -= alpha * db1[j];
      for (let i = 0; i < this.inputSize; i++) {
        this.W1[i][j] -= alpha * dW1[i][j];
      }
    }
    for (let k = 0; k < this.outputSize; k++) {
      this.b2[k] -= alpha * db2[k];
      for (let j = 0; j < this.hiddenSize; j++) {
        this.W2[j][k] -= alpha * dW2[j][k];
      }
    }

    return loss;
  }

  copy(): SimpleMLP {
    const net = new SimpleMLP(this.inputSize, this.hiddenSize, this.outputSize);
    net.W1 = this.W1.map((row) => [...row]);
    net.b1 = [...this.b1];
    net.W2 = this.W2.map((row) => [...row]);
    net.b2 = [...this.b2];
    return net;
  }
}

function randomMatrix(rows: number, cols: number, scale: number): number[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
  );
}

// ---------------------------------------------------------------------------
// DQN on the GridWorld using coordinate features
// ---------------------------------------------------------------------------

export function dqnGridWorld(
  config: GridWorldConfig,
  options: DQNOptions
): {
  qHistory: number[][][];
  lossHistory: number[];
  finalReplaySize: number;
} {
  const {
    hiddenSize,
    alpha,
    epsilon,
    gamma = config.gamma,
    batchSize,
    replayCapacity,
    targetUpdateInterval,
    episodes,
    maxSteps,
  } = options;

  const numStates = config.rows * config.cols;
  const numActions = 5;
  const featureDim = 3; // [1, rNorm, cNorm]

  const mainNet = new SimpleMLP(featureDim, hiddenSize, numActions);
  const targetNet = mainNet.copy();
  const replay: Transition[] = [];

  const qHistory: number[][][] = [];
  const lossHistory: number[] = [];
  let trainStepCount = 0;

  function stateFeatures(state: number): number[] {
    return coordinateFeatures(state, config);
  }

  function networkQ(net: SimpleMLP, state: number): number[] {
    return net.forward(stateFeatures(state)).out;
  }

  for (let ep = 0; ep < episodes; ep++) {
    let state = config.startState;

    for (let stepIdx = 0; stepIdx < maxSteps; stepIdx++) {
      if (isTerminal(state, config)) break;

      // ε-greedy action selection from main network
      let action: Action;
      const qValues = networkQ(mainNet, state);
      if (Math.random() < epsilon) {
        action = Math.floor(Math.random() * numActions) as Action;
      } else {
        const maxQ = Math.max(...qValues);
        const best = qValues
          .map((q, i) => ({ q, i }))
          .filter(({ q }) => Math.abs(q - maxQ) < 1e-6)
          .map(({ i }) => i);
        action = best[Math.floor(Math.random() * best.length)] as Action;
      }

      const result = step(state, action, config);
      replay.push({ state, action, reward: result.reward, nextState: result.nextState, done: result.done });
      if (replay.length > replayCapacity) replay.shift();

      // Replay training
      if (replay.length >= batchSize) {
        const batch = sampleReplay(replay, batchSize);
        let totalLoss = 0;
        for (const trans of batch) {
          const qNext = networkQ(targetNet, trans.nextState);
          const maxQNext = Math.max(...qNext);
          const target = trans.done ? trans.reward : trans.reward + gamma * maxQNext;
          const loss = mainNet.trainStep(stateFeatures(trans.state), trans.action, target, alpha);
          totalLoss += loss;
        }
        lossHistory.push(totalLoss / batch.length);
        trainStepCount++;

        if (trainStepCount % targetUpdateInterval === 0) {
          // Copy main network weights to target network
          targetNet.W1 = mainNet.W1.map((row) => [...row]);
          targetNet.b1 = [...mainNet.b1];
          targetNet.W2 = mainNet.W2.map((row) => [...row]);
          targetNet.b2 = [...mainNet.b2];
        }
      }

      state = result.nextState;
      if (result.done) break;
    }

    const qTable: number[][] = [];
    for (let s = 0; s < numStates; s++) {
      qTable.push(networkQ(mainNet, s));
    }
    qHistory.push(qTable);
  }

  return { qHistory, lossHistory, finalReplaySize: replay.length };
}

function sampleReplay<T>(buffer: T[], n: number): T[] {
  const sample: T[] = [];
  const copy = [...buffer];
  for (let i = 0; i < n && copy.length > 0; i++) {
    const idx = Math.floor(Math.random() * copy.length);
    sample.push(copy[idx]);
    copy.splice(idx, 1);
  }
  return sample;
}

// ---------------------------------------------------------------------------
// Action-value function approximation
// ---------------------------------------------------------------------------

export type ActionValueFeatureMode = 'onehot' | 'shared';

function distanceStateFeatures(state: number, config: GridWorldConfig): number[] {
  const { row, col } = { row: Math.floor(state / config.cols), col: state % config.cols };
  const rNorm = config.rows > 1 ? (row / (config.rows - 1)) * 2 - 1 : 0;
  const cNorm = config.cols > 1 ? (col / (config.cols - 1)) * 2 - 1 : 0;
  const { row: tRow, col: tCol } = {
    row: Math.floor(config.targetState / config.cols),
    col: config.targetState % config.cols,
  };
  const distanceToTarget = Math.sqrt((row - tRow) ** 2 + (col - tCol) ** 2);
  const maxDist = Math.sqrt((config.rows - 1) ** 2 + (config.cols - 1) ** 2);
  const isForbidden = config.forbiddenStates.includes(state) ? 1 : 0;
  return [1, rNorm, cNorm, distanceToTarget / Math.max(1, maxDist), isForbidden];
}

export function actionValueFeatureDim(
  config: GridWorldConfig,
  mode: ActionValueFeatureMode
): number {
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
  gamma?: number;
  episodes: number;
  maxSteps: number;
  featureMode: ActionValueFeatureMode;
  algorithm: 'sarsa' | 'qlearning';
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

export function actionValueFA(
  config: GridWorldConfig,
  options: ActionValueFAOptions
): {
  qHistory: number[][][];
  weightsHistory: number[][];
  lastUpdate: LastFAUpdate | null;
} {
  const { alpha, epsilon, gamma = config.gamma, episodes, maxSteps, featureMode, algorithm } = options;
  const numStates = config.rows * config.cols;
  const numActions = 5;
  const featureDim = actionValueFeatureDim(config, featureMode);
  let w = new Array(featureDim).fill(0);

  const qHistory: number[][][] = [];
  const weightsHistory: number[][] = [];
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
    return best[Math.floor(Math.random() * best.length)] as Action;
  }

  function sampleEpsilonGreedy(state: number): Action {
    if (Math.random() < epsilon) {
      return Math.floor(Math.random() * numActions) as Action;
    }
    return greedyAction(state);
  }

  for (let ep = 0; ep < episodes; ep++) {
    let state = config.startState;
    let action = sampleEpsilonGreedy(state);

    for (let stepIdx = 0; stepIdx < maxSteps; stepIdx++) {
      if (isTerminal(state, config)) break;
      const result = step(state, action, config);
      const phi = actionValueFeatures(state, action, config, featureMode);
      const prediction = dot(phi, w);

      let target = result.reward;
      let nextAction: Action | undefined;
      if (!result.done) {
        if (algorithm === 'sarsa') {
          nextAction = sampleEpsilonGreedy(result.nextState);
          target += gamma * qValue(result.nextState, nextAction);
        } else {
          const maxQ = Math.max(
            ...Array.from({ length: numActions }, (_, a) => qValue(result.nextState, a as Action))
          );
          target += gamma * maxQ;
        }
      }

      const tdError = target - prediction;
      const update = alpha * tdError;
      const weightBefore = w.map((x) => x);
      for (let i = 0; i < featureDim; i++) {
        w[i] += update * phi[i];
      }
      const weightChange = Math.sqrt(
        w.reduce((sum, wi, i) => sum + (wi - weightBefore[i]) ** 2, 0)
      );

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
      action = algorithm === 'sarsa' && nextAction !== undefined ? nextAction : sampleEpsilonGreedy(state);
      if (result.done) break;
    }

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
  }

  return { qHistory, weightsHistory, lastUpdate };
}
