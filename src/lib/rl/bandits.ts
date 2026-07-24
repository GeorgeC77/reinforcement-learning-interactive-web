/**
 * Multi-armed bandit environment and exploration strategies.
 *
 * Used by the chapter 11 demos to compare ε-greedy, UCB1, and Boltzmann
 * (softmax) exploration on the same reward distributions. All randomness is
 * driven by a caller-supplied seeded rng so experiments are reproducible.
 */

import { mulberry32 } from './stochasticApproximation';

export interface BanditArm {
  mean: number;
  std: number;
}

export type BanditStrategy = 'eps-greedy' | 'ucb1' | 'softmax';

export interface BanditOptions {
  strategy: BanditStrategy;
  steps: number;
  epsilon?: number;
  /** Boltzmann temperature τ for the softmax strategy. */
  tau?: number;
  /** Exploration constant c for UCB1. */
  ucbC?: number;
  seed?: number;
}

export interface BanditStep {
  t: number;
  action: number;
  reward: number;
  estimates: number[];
  /** Cumulative reward up to and including this step. */
  cumulativeReward: number;
  /** Cumulative regret: Σ (μ* − μ_{a_t}). */
  regret: number;
  /** Running fraction of optimal-action selections. */
  optimalRate: number;
}

export interface BanditResult {
  arms: BanditArm[];
  optimalAction: number;
  steps: BanditStep[];
  finalEstimates: number[];
  actionCounts: number[];
  /** Final cumulative regret. */
  totalRegret: number;
}

/** Box–Muller sample from N(mean, std²). */
function sampleNormal(mean: number, std: number, rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + std * z;
}

function selectAction(
  q: number[],
  counts: number[],
  t: number,
  strategy: BanditStrategy,
  epsilon: number,
  tau: number,
  ucbC: number,
  rng: () => number
): number {
  const k = q.length;
  if (strategy === 'eps-greedy') {
    if (rng() < epsilon) return Math.floor(rng() * k);
    let best = 0;
    for (let a = 1; a < k; a++) if (q[a] > q[best]) best = a;
    return best;
  }
  if (strategy === 'ucb1') {
    // Pull each arm once before applying the UCB rule.
    for (let a = 0; a < k; a++) if (counts[a] === 0) return a;
    const logT = Math.log(t);
    let best = 0;
    let bestScore = -Infinity;
    for (let a = 0; a < k; a++) {
      const score = q[a] + ucbC * Math.sqrt(logT / counts[a]);
      if (score > bestScore) {
        bestScore = score;
        best = a;
      }
    }
    return best;
  }
  // softmax (Boltzmann)
  const scaled = q.map((v) => v / tau);
  const m = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - m));
  const sum = exps.reduce((s, x) => s + x, 0);
  const r = rng() * sum;
  let cum = 0;
  for (let a = 0; a < k; a++) {
    cum += exps[a];
    if (r <= cum) return a;
  }
  return k - 1;
}

/**
 * Run a bandit simulation and return the full step history.
 * Estimates use sample-average updates (constant structure, not constant-α).
 */
export function runBandit(arms: BanditArm[], options: BanditOptions): BanditResult {
  const { strategy, steps, epsilon = 0.1, tau = 0.5, ucbC = 1.5, seed = 1 } = options;
  const rng = mulberry32(seed);
  const k = arms.length;
  const means = arms.map((a) => a.mean);
  const maxMean = Math.max(...means);
  const optimalAction = means.indexOf(maxMean);

  const q = new Array(k).fill(0);
  const counts = new Array(k).fill(0);
  const actionCounts = new Array(k).fill(0);
  const history: BanditStep[] = [];

  let cumulativeReward = 0;
  let regret = 0;
  let optimalPicks = 0;

  for (let t = 1; t <= steps; t++) {
    const action = selectAction(q, counts, t, strategy, epsilon, tau, ucbC, rng);
    const reward = sampleNormal(arms[action].mean, arms[action].std, rng);
    counts[action]++;
    actionCounts[action]++;
    q[action] += (reward - q[action]) / counts[action];

    cumulativeReward += reward;
    regret += maxMean - arms[action].mean;
    if (action === optimalAction) optimalPicks++;

    history.push({
      t,
      action,
      reward,
      estimates: [...q],
      cumulativeReward,
      regret,
      optimalRate: optimalPicks / t,
    });
  }

  return {
    arms: arms.map((a) => ({ ...a })),
    optimalAction,
    steps: history,
    finalEstimates: [...q],
    actionCounts,
    totalRegret: regret,
  };
}
