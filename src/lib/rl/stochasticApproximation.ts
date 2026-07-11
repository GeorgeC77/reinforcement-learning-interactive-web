/**
 * Stochastic approximation utilities for Chapter 6.
 *
 * All functions support an optional seed so that different algorithms can be
 * compared on the exact same random samples / noise sequence.
 */

import {
  type GridWorldConfig,
  type Policy,
  type Action,
  step,
  sampleAction,
  isTerminal,
} from './gridworld';

// ---------------------------------------------------------------------------
// Seeded random number generation
// ---------------------------------------------------------------------------

export function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return function () {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function makeRng(seed: number): () => number {
  return mulberry32(seed);
}

export function normalRandom(rng: () => number, mean: number, std: number): number {
  // Box-Muller
  const u1 = rng();
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + std * z;
}

export function generateNormalSamples(
  n: number,
  mean: number,
  std: number,
  seed: number
): number[] {
  const rng = makeRng(seed);
  return Array.from({ length: n }, () => normalRandom(rng, mean, std));
}

// ---------------------------------------------------------------------------
// Tab 1: batch vs incremental mean
// ---------------------------------------------------------------------------

export interface MeanComparisonStep {
  n: number;
  sample: number;
  batchMean: number;
  incrementalMean: number;
  predictionError: number;
  stepSize: number;
}

export function batchVsIncrementalMean(samples: number[]): MeanComparisonStep[] {
  const history: MeanComparisonStep[] = [];
  let incrementalMean = 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    const n = i + 1;
    sum += samples[i];
    const batchMean = sum / n;
    const predictionError = samples[i] - incrementalMean;
    const stepSize = 1 / n;
    incrementalMean = incrementalMean + stepSize * predictionError;
    history.push({
      n,
      sample: samples[i],
      batchMean,
      incrementalMean,
      predictionError,
      stepSize,
    });
  }
  return history;
}

// ---------------------------------------------------------------------------
// Tab 2: Robbins-Monro root finding
// ---------------------------------------------------------------------------

export type RMFunction = 'linear' | 'tanh';

export interface RMStep {
  k: number;
  w: number;
  alpha: number;
  trueG: number;
  noisyG: number;
  update: number;
  wNext: number;
}

function evaluateG(w: number, wStar: number, gName: RMFunction): number {
  const delta = w - wStar;
  if (gName === 'tanh') return Math.tanh(delta);
  return delta;
}

export function robbinsMonroSequence(
  wStar: number,
  initialW: number,
  alphas: number[],
  noiseStd: number,
  gName: RMFunction,
  seed: number
): RMStep[] {
  const rng = makeRng(seed);
  const history: RMStep[] = [];
  let w = initialW;
  for (let k = 0; k < alphas.length; k++) {
    const alpha = alphas[k];
    const trueG = evaluateG(w, wStar, gName);
    const noise = normalRandom(rng, 0, noiseStd);
    const noisyG = trueG + noise;
    const update = -alpha * noisyG;
    const wNext = w + update;
    history.push({ k, w, alpha, trueG, noisyG, update, wNext });
    w = wNext;
  }
  return history;
}

// ---------------------------------------------------------------------------
// Tab 3: step-size condition checker
// ---------------------------------------------------------------------------

export interface StepSizeCondition {
  label: string;
  description: string;
  valid: boolean;
}

export function checkStepSizeCondition(p: number): StepSizeCondition {
  if (p <= 0.5) {
    return {
      label: 'p ≤ 0.5',
      description:
        'Σ α_k 发散，Σ α_k² 发散。不满足经典 Robbins-Monro 条件：噪声难以充分消退。',
      valid: false,
    };
  }
  if (p <= 1) {
    return {
      label: '0.5 < p ≤ 1',
      description:
        'Σ α_k 发散，Σ α_k² 收敛。满足经典 Robbins-Monro 步长条件。',
      valid: true,
    };
  }
  return {
    label: 'p > 1',
    description:
      'Σ α_k 收敛，Σ α_k² 收敛。不满足经典条件：累计更新量可能不足，难以保证收敛。',
      valid: false,
  };
}

export function powerStepSizes(n: number, power: number, offset = 1): number[] {
  return Array.from({ length: n }, (_, i) => 1 / Math.pow(i + offset, power));
}

// ---------------------------------------------------------------------------
// Tab 4: BGD / MBGD / SGD for mean estimation
// ---------------------------------------------------------------------------

export type GDMode = 'bgd' | 'mbgd' | 'sgd';

export interface GDStep {
  epoch: number;
  step: number;
  w: number;
  loss: number;
  gradient: number;
  batchSize: number;
  samplesProcessed: number;
  gradientVariance: number;
}

function shuffleArray<T>(arr: T[], rng: () => number): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function meanEstimationGradientDescent(
  dataset: number[],
  initialW: number,
  mode: GDMode,
  batchSize: number,
  alpha: number,
  epochs: number,
  seed: number
): GDStep[] {
  const rng = makeRng(seed);
  const n = dataset.length;
  const effectiveBatchSize = mode === 'bgd' ? n : mode === 'sgd' ? 1 : Math.min(batchSize, n);
  const history: GDStep[] = [];
  let w = initialW;
  let samplesProcessed = 0;
  let step = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const shuffled = shuffleArray(dataset, rng);

    if (mode === 'bgd') {
      const grads = dataset.map((x) => w - x);
      const gradient = grads.reduce((a, b) => a + b, 0) / n;
      const gradientVariance =
        grads.reduce((sum, g) => sum + Math.pow(g - gradient, 2), 0) / n;
      const loss = grads.reduce((sum, g) => sum + g * g, 0) / (2 * n);
      w -= alpha * gradient;
      samplesProcessed += n;
      step++;
      history.push({
        epoch,
        step,
        w,
        loss,
        gradient,
        batchSize: n,
        samplesProcessed,
        gradientVariance,
      });
      continue;
    }

    for (let i = 0; i < n; i += effectiveBatchSize) {
      const batch = shuffled.slice(i, i + effectiveBatchSize);
      const grads = batch.map((x) => w - x);
      const gradient = grads.reduce((a, b) => a + b, 0) / batch.length;
      const gradientVariance =
        grads.reduce((sum, g) => sum + Math.pow(g - gradient, 2), 0) / batch.length;
      const loss = grads.reduce((sum, g) => sum + g * g, 0) / (2 * batch.length);
      w -= alpha * gradient;
      samplesProcessed += batch.length;
      step++;
      history.push({
        epoch,
        step,
        w,
        loss,
        gradient,
        batchSize: batch.length,
        samplesProcessed,
        gradientVariance,
      });
    }
  }

  return history;
}

// ---------------------------------------------------------------------------
// Tab 5: SA -> TD bridge
// ---------------------------------------------------------------------------

export interface TDBridgeStep {
  state: number;
  action: Action;
  reward: number;
  nextState: number;
  done: boolean;
  vOld: number;
  vNextOld: number;
  tdTarget: number;
  tdError: number;
  vNew: number;
}

export function tdBridgeStep(
  config: GridWorldConfig,
  policy: Policy,
  v: number[],
  alpha: number,
  seed: number
): { step: TDBridgeStep; vNew: number[] } {
  const rng = makeRng(seed);
  const numStates = config.rows * config.cols;
  let state: number;
  // Try to avoid starting in terminal state
  for (let attempts = 0; attempts < 10; attempts++) {
    state = Math.floor(rng() * numStates);
    if (!isTerminal(state, config)) break;
  }
  state = state!;
  const action = sampleAction(policy[state]);
  const result = step(state, action, config);

  const vOld = v[state];
  const vNextOld = result.done ? 0 : v[result.nextState];
  const tdTarget = result.reward + config.gamma * vNextOld;
  const tdError = tdTarget - vOld;
  const updated = vOld + alpha * tdError;

  const vNew = [...v];
  vNew[state] = updated;

  return {
    step: {
      state,
      action,
      reward: result.reward,
      nextState: result.nextState,
      done: result.done,
      vOld,
      vNextOld,
      tdTarget,
      tdError,
      vNew: updated,
    },
    vNew,
  };
}
