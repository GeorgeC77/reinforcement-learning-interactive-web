/**
 * Stochastic approximation utilities for Chapter 6.
 *
 * All randomness is driven by a seeded mulberry32 generator so that different
 * algorithms, trajectories and noise sequences can be reproduced exactly.
 */

import {
  type GridWorldConfig,
  type Policy,
  type Action,
  step,
  sampleActionWithRng,
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
  // Box-Muller.  Guard against u1 == 0 so log(0) cannot occur.
  const u1 = Math.max(rng(), Number.EPSILON);
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

export function evaluateGPrime(w: number, wStar: number, gName: RMFunction): number {
  const delta = w - wStar;
  if (gName === 'tanh') {
    // derivative of tanh(x) = sech^2(x) = 1 / cosh^2(x)
    const cosh = (Math.exp(delta) + Math.exp(-delta)) / 2;
    return 1 / (cosh * cosh);
  }
  return 1;
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
        'Σ α_k 发散，Σ α_k² 发散。不满足经典 Robbins-Monro 步长条件：噪声难以充分消退。',
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

export function partialSumCondition(
  n: number,
  power: number
): { sum: number; sumSquares: number; diverges: boolean } {
  let sum = 0;
  let sumSquares = 0;
  for (let i = 1; i <= n; i++) {
    const a = 1 / Math.pow(i, power);
    sum += a;
    sumSquares += a * a;
  }
  return {
    sum,
    sumSquares,
    diverges: power <= 1,
  };
}

// ---------------------------------------------------------------------------
// Tab 4: BGD / MBGD / SGD for mean estimation
// ---------------------------------------------------------------------------

export type GDMode = 'bgd' | 'mbgd' | 'sgd';

export interface GDStep {
  epoch: number;
  step: number;
  wBefore: number;
  wAfter: number;
  batchGradient: number;
  fullGradient: number;
  gradientNoise: number;
  squaredGradientNoise: number;
  batchLossBefore: number;
  fullLossBefore: number;
  fullLossAfter: number;
  batchSize: number;
  samplesProcessed: number;
}

export function fullObjective(dataset: number[], w: number): number {
  return dataset.reduce((sum, x) => sum + 0.5 * (w - x) ** 2, 0) / dataset.length;
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
      const wBefore = w;
      const fullGradient = dataset.reduce((sum, x) => sum + (wBefore - x), 0) / n;
      const batchGradient = fullGradient;
      const gradientNoise = batchGradient - fullGradient;
      const squaredGradientNoise = gradientNoise ** 2;
      const batchLossBefore = fullObjective(dataset, wBefore);
      const fullLossBefore = batchLossBefore;
      const wAfter = wBefore - alpha * batchGradient;
      const fullLossAfter = fullObjective(dataset, wAfter);
      w = wAfter;
      samplesProcessed += n;
      step++;
      history.push({
        epoch,
        step,
        wBefore,
        wAfter,
        batchGradient,
        fullGradient,
        gradientNoise,
        squaredGradientNoise,
        batchLossBefore,
        fullLossBefore,
        fullLossAfter,
        batchSize: n,
        samplesProcessed,
      });
      continue;
    }

    for (let i = 0; i < n; i += effectiveBatchSize) {
      const batch = shuffled.slice(i, i + effectiveBatchSize);
      const wBefore = w;
      const fullGradient = dataset.reduce((sum, x) => sum + (wBefore - x), 0) / n;
      const batchGradient = batch.reduce((sum, x) => sum + (wBefore - x), 0) / batch.length;
      const gradientNoise = batchGradient - fullGradient;
      const squaredGradientNoise = gradientNoise ** 2;
      const batchLossBefore = batch.reduce((sum, x) => sum + 0.5 * (wBefore - x) ** 2, 0) / batch.length;
      const fullLossBefore = fullObjective(dataset, wBefore);
      const wAfter = wBefore - alpha * batchGradient;
      const fullLossAfter = fullObjective(dataset, wAfter);
      w = wAfter;
      samplesProcessed += batch.length;
      step++;
      history.push({
        epoch,
        step,
        wBefore,
        wAfter,
        batchGradient,
        fullGradient,
        gradientNoise,
        squaredGradientNoise,
        batchLossBefore,
        fullLossBefore,
        fullLossAfter,
        batchSize: batch.length,
        samplesProcessed,
      });
    }
  }

  return history;
}

export interface FixedWNoiseEstimate {
  w: number;
  meanBatchGradient: number;
  varianceOfBatchGradients: number;
}

/**
 * Estimate Var[g_B(w)] for a fixed w by repeatedly sampling batches.
 * Useful for explaining why SGD has high gradient noise while BGD has none.
 */
export function estimateBatchGradientVariance(
  dataset: number[],
  w: number,
  mode: Exclude<GDMode, 'bgd'>,
  batchSize: number,
  numSamples: number,
  seed: number
): FixedWNoiseEstimate {
  const rng = makeRng(seed);
  const n = dataset.length;
  const effectiveBatchSize = mode === 'sgd' ? 1 : Math.min(batchSize, n);
  const gradients: number[] = [];

  for (let i = 0; i < numSamples; i++) {
    const shuffled = shuffleArray(dataset, rng);
    const batch = shuffled.slice(0, effectiveBatchSize);
    const g = batch.reduce((sum, x) => sum + (w - x), 0) / batch.length;
    gradients.push(g);
  }

  const mean = gradients.reduce((a, b) => a + b, 0) / gradients.length;
  const variance =
    gradients.reduce((sum, g) => sum + (g - mean) ** 2, 0) / gradients.length;

  return { w, meanBatchGradient: mean, varianceOfBatchGradients: variance };
}

export function movingAverage(values: number[], window: number): number[] {
  const result: number[] = [];
  const w = Math.max(1, Math.min(window, values.length));
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= w) sum -= values[i - w];
    result.push(sum / Math.min(i + 1, w));
  }
  return result;
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
  currentState: number,
  rng: () => number
): { step: TDBridgeStep; vNew: number[]; nextCurrentState: number } {
  const state = currentState;
  const action = sampleActionWithRng(policy[state], rng);
  const result = step(state, action, config);

  const vOld = v[state];
  const vNextOld = result.done ? 0 : v[result.nextState];
  const tdTarget = result.reward + config.gamma * vNextOld;
  const tdError = tdTarget - vOld;
  const updated = vOld + alpha * tdError;

  const vNew = [...v];
  vNew[state] = updated;

  const nextCurrentState = result.done ? config.startState : result.nextState;

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
    nextCurrentState,
  };
}

// ---------------------------------------------------------------------------
// Tab 6: Dvoretzky convergence theorem demonstration
// ---------------------------------------------------------------------------

export interface DvoretzkyStep {
  k: number;
  delta: number;
  alpha: number;
  beta: number;
  noise: number;
}

export function dvoretzkyErrorSequence(
  initialDelta: number,
  alphas: number[],
  betas: number[],
  noiseStd: number,
  seed: number
): DvoretzkyStep[] {
  const rng = makeRng(seed);
  const history: DvoretzkyStep[] = [];
  let delta = initialDelta;
  for (let k = 0; k < alphas.length; k++) {
    const alpha = alphas[k];
    const beta = betas[k];
    const noise = normalRandom(rng, 0, noiseStd);
    delta = (1 - alpha) * delta + beta * noise;
    history.push({ k, delta, alpha, beta, noise });
  }
  return history;
}
