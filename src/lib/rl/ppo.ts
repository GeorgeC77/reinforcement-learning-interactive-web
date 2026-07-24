/**
 * PPO clipped-surrogate intuition helpers.
 *
 * Provides the exact clipped/unclipped surrogate objectives and a minimal
 * one-state two-action trainer that contrasts how unclipped ratio updates can
 * overshoot while the PPO clip keeps the policy ratio inside [1-ε, 1+ε].
 * Deterministic (exact expectations, no sampling noise) so the didactic
 * contrast is crisp; no rng needed.
 */

export interface ClipObjective {
  unclipped: number;
  clipped: number;
  /** clip(r, 1-ε, 1+ε) */
  ratioClipped: number;
}

/** Single-sample clipped surrogate: min(r·A, clip(r,1-ε,1+ε)·A). */
export function clipRatioObjective(r: number, A: number, epsilon: number): ClipObjective {
  const lo = 1 - epsilon;
  const hi = 1 + epsilon;
  const ratioClipped = Math.min(hi, Math.max(lo, r));
  return {
    unclipped: r * A,
    clipped: Math.min(r * A, ratioClipped * A),
    ratioClipped,
  };
}

/** Curve of the two objectives as a function of the ratio r (for plotting). */
export function clipObjectiveCurve(
  A: number,
  epsilon: number,
  rMin = 0,
  rMax = 2,
  points = 81
): { r: number; unclipped: number; clipped: number }[] {
  const data: { r: number; unclipped: number; clipped: number }[] = [];
  for (let i = 0; i < points; i++) {
    const r = rMin + ((rMax - rMin) * i) / (points - 1);
    const o = clipRatioObjective(r, A, epsilon);
    data.push({ r, unclipped: o.unclipped, clipped: o.clipped });
  }
  return data;
}

// ---------------------------------------------------------------------------
// Minimal one-state two-action PPO trainer
// ---------------------------------------------------------------------------

export interface PpoToyOptions {
  /** Mean rewards of the two actions (action 1 is better when mu1 > mu0). */
  mu0?: number;
  mu1?: number;
  /** Learning rate on the single logit θ (p1 = sigmoid(θ)). */
  alpha: number;
  /** PPO clip range ε. */
  epsilon: number;
  iterations: number;
  /** 'clipped' uses the PPO surrogate, 'unclipped' maximizes r·A directly. */
  mode: 'clipped' | 'unclipped';
  /** Initial logit. */
  theta0?: number;
}

export interface PpoToyStep {
  iteration: number;
  theta: number;
  p1: number;
  /** Exact expected surrogate objective before this update. */
  objective: number;
  /** Ratio r1 = p1_new/p1_old that this update would imply at the optimum. */
  maxRatio: number;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Exact expected surrogate over the two actions:
 *   L(θ) = p0_old · f(r0(θ)) · A0 + p1_old · f(r1(θ)) · A1,
 * where f(r) = r (unclipped) or min(r·A, clip(r)·A) per action (clipped).
 * The advantage is exact: A_a = μ_a − V_old.
 */
export function runPpoToy(options: PpoToyOptions): PpoToyStep[] {
  const { mu0 = 0, mu1 = 1, alpha, epsilon, iterations, mode, theta0 = 0 } = options;
  let theta = theta0;
  const history: PpoToyStep[] = [];

  for (let k = 0; k < iterations; k++) {
    // "Old" policy is the current one before this update (on-policy batch).
    const p1Old = sigmoid(theta);
    const p0Old = 1 - p1Old;
    const vOld = p0Old * mu0 + p1Old * mu1;
    const A0 = mu0 - vOld;
    const A1 = mu1 - vOld;

    // Exact expected surrogate as a function of the new probability q = p1_new.
    function objective(q: number): number {
      const r1 = q / p1Old;
      const r0 = (1 - q) / p0Old;
      if (mode === 'unclipped') {
        return p0Old * r0 * A0 + p1Old * r1 * A1;
      }
      return p0Old * clipRatioObjective(r0, A0, epsilon).clipped + p1Old * clipRatioObjective(r1, A1, epsilon).clipped;
    }

    // Gradient-free update: sample a small step along the objective's ascent
    // direction via finite differences on q = sigmoid(θ).
    const h = 1e-5;
    const q = sigmoid(theta);
    const grad = (objective(sigmoid(theta + h)) - objective(sigmoid(theta - h))) / (2 * h);
    const objBefore = objective(q);
    const maxRatio = Math.max(q / p1Old, (1 - q) / p0Old);
    theta += alpha * grad;

    history.push({
      iteration: k,
      theta,
      p1: sigmoid(theta),
      objective: objBefore,
      maxRatio,
    });
  }

  return history;
}
