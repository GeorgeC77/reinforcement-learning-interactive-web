import { runBandit, type BanditArm } from './bandits';

function assert(cond: boolean, msg: string): asserts cond {
  if (!cond) throw new Error(`Test failed: ${msg}`);
}

const ARMS: BanditArm[] = [
  { mean: 0.2, std: 1 },
  { mean: 0.5, std: 1 },
  { mean: 1.0, std: 1 },
];

export function runBanditTests() {
  // 1. Reproducibility with the same seed.
  const a = runBandit(ARMS, { strategy: 'eps-greedy', steps: 200, seed: 42 });
  const b = runBandit(ARMS, { strategy: 'eps-greedy', steps: 200, seed: 42 });
  assert(
    a.steps.every((s, i) => s.action === b.steps[i].action && s.reward === b.steps[i].reward),
    'same seed should produce identical trajectories'
  );

  // 2. Estimates converge toward the true means with enough samples.
  const long = runBandit(ARMS, { strategy: 'eps-greedy', steps: 5000, epsilon: 0.1, seed: 7 });
  const maxErr = Math.max(...long.finalEstimates.map((est, i) => Math.abs(est - ARMS[i].mean)));
  assert(maxErr < 0.2, `estimates should converge to true means (max err ${maxErr.toFixed(3)})`);

  // 3. UCB1 pulls every arm at least once in the first k steps.
  const ucb = runBandit(ARMS, { strategy: 'ucb1', steps: 100, seed: 3 });
  for (let arm = 0; arm < ARMS.length; arm++) {
    assert(ucb.steps[arm].action === arm, 'UCB1 should initialize by pulling each arm once');
  }

  // 4. All strategies learn to favor the optimal arm (optimal rate rises).
  for (const strategy of ['eps-greedy', 'ucb1', 'softmax'] as const) {
    const r = runBandit(ARMS, { strategy, steps: 2000, seed: 11 });
    const late = r.steps.slice(-500);
    const lateRate = late.filter((s) => s.action === r.optimalAction).length / late.length;
    assert(lateRate > 0.5, `${strategy} should favor the optimal arm late in training (rate ${lateRate.toFixed(2)})`);
  }

  // 5. Regret accumulates and is non-decreasing.
  const r = runBandit(ARMS, { strategy: 'eps-greedy', steps: 100, seed: 5 });
  for (let i = 1; i < r.steps.length; i++) {
    assert(r.steps[i].regret >= r.steps[i - 1].regret, 'regret should be non-decreasing');
  }

  // 6. Greedy-with-zero-epsilon picks only the current best arm after step 1.
  const greedy = runBandit(ARMS, { strategy: 'eps-greedy', steps: 50, epsilon: 0, seed: 9 });
  const first = greedy.steps[0].action;
  assert(
    greedy.steps.every((s) => s.action === first),
    'pure greedy (ε=0) should stick to one arm'
  );

  console.log('All bandit tests passed.');
}
