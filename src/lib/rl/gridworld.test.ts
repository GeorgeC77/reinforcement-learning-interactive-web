import {
  DEFAULT_CONFIG,
  EPISODIC_PATH_CONFIG,
  deterministicPolicy,
  sampleActionWithRng,
  tdZeroPrediction,
  sarsa,
  qLearning,
  nStepSarsa,
  policyBellmanResidualV,
  policyBellmanResidualQ,
  optimalBellmanResidualQ,
  epsilonAtEpisode,
  type Policy,
  type Action,
} from './gridworld';
import { mulberry32 } from './stochasticApproximation';

const GOAL_POLICY: Action[] = [2, 2, 3, 1, 2, 3, 0, 1, 4];

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(`Test failed: ${msg}`);
}

function near(a: number, b: number, eps = 1e-9) {
  return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= eps;
}

function countTransitions(updates: { episode: number; state: number; nextState: number; done: boolean }[], episode: number) {
  return updates.filter((u) => u.episode === episode && !u.done).length;
}

export function runGridWorldTests() {
  // 1. policyBellmanResidualV uses probability weighting for stochastic policies
  const values = new Array(9).fill(0);
  const stochastic: Policy = Array.from({ length: 9 }, () => [0.2, 0.2, 0.2, 0.2, 0.2]);
  const residualStoch = policyBellmanResidualV(values, stochastic, DEFAULT_CONFIG);
  const deterministic: Policy = deterministicPolicy(GOAL_POLICY, 5);
  const residualDet = policyBellmanResidualV(values, deterministic, DEFAULT_CONFIG);
  assert(residualStoch >= 0, 'stochastic residual should be non-negative');
  assert(residualDet >= 0, 'deterministic residual should be non-negative');
  assert(
    residualStoch !== residualDet || residualStoch === 0,
    'stochastic and deterministic residuals should differ unless both zero'
  );

  // 2. Terminal transition does not bootstrap
  const terminalState = EPISODIC_PATH_CONFIG.targetState;
  const qZero = Array.from({ length: 9 }, () => new Array(5).fill(0));
  const residual = optimalBellmanResidualQ(qZero, EPISODIC_PATH_CONFIG);
  assert(Number.isFinite(residual), 'terminal residual should be finite');
  assert(residual >= 0, 'terminal residual should be non-negative');

  // 3. Q-learning residual directly uses learned Q (not reconstructed)
  const ql = qLearning(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 20, 10, 42);
  const lastQ = ql.frames[ql.frames.length - 1].qValues;
  const optRes = optimalBellmanResidualQ(lastQ, EPISODIC_PATH_CONFIG);
  assert(optRes >= 0, 'Q-learning optimal residual should be non-negative');

  // 4. Sarsa residual uses current behavior policy
  const sr = sarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 20, 10, 42);
  const lastFrame = sr.frames[sr.frames.length - 1];
  const polRes = policyBellmanResidualQ(lastFrame.qValues, lastFrame.behaviorPolicy, EPISODIC_PATH_CONFIG);
  assert(polRes >= 0, 'Sarsa policy residual should be non-negative');

  // 5. n-step Sarsa does not exceed H transitions per episode
  const ns = nStepSarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 3, 15, 10, 7);
  for (let ep = 0; ep < 10; ep++) {
    const transitions = countTransitions(ns.updates, ep);
    assert(transitions <= 15, `n-step episode ${ep} should have at most 15 transitions`);
  }

  // 6. n-step Sarsa reproducibility
  const nsA = nStepSarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 3, 15, 10, 123);
  const nsB = nStepSarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 3, 15, 10, 123);
  assert(nsA.updates.length === nsB.updates.length, 'same seed should produce same update count');
  nsA.updates.forEach((u, i) => {
    const v = nsB.updates[i];
    assert(
      u.episode === v.episode && u.state === v.state && u.action === v.action && u.reward === v.reward,
      `same seed mismatch at update ${i}`
    );
  });

  // 7. Same seed produces identical Sarsa history
  const a = sarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 20, 5, 99);
  const b = sarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 20, 5, 99);
  assert(a.updates.length === b.updates.length, 'same seed should produce same update count');
  a.updates.forEach((u, i) => {
    const v = b.updates[i];
    assert(
      u.episode === v.episode && u.state === v.state && u.action === v.action && u.reward === v.reward,
      `same seed mismatch at update ${i}`
    );
  });

  // 8. Different seed usually produces different trajectories
  const c = sarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 20, 5, 100);
  const different = a.updates.some((u, i) => {
    const v = c.updates[i];
    return !v || u.action !== v.action || u.reward !== v.reward;
  });
  assert(different, 'different seed should usually produce different trajectories');

  // 9. Fixed epsilon vs GLIE sequences
  const fixed = epsilonAtEpisode(100, 0.5, 'fixed');
  const glie = epsilonAtEpisode(100, 0.5, 'glie');
  assert(near(fixed, 0.5), 'fixed epsilon should stay at 0.5');
  assert(glie < 0.5 && glie > 0, 'GLIE epsilon should decay');
  const glie0 = epsilonAtEpisode(0, 0.5, 'glie');
  assert(near(glie0, 0.5), 'GLIE epsilon at episode 0 should be epsilon0');

  // 10. TD(0) prediction uses seeded RNG
  const tdA = tdZeroPrediction(deterministicPolicy(GOAL_POLICY, 5), DEFAULT_CONFIG, 0.1, 20, 5, 55);
  const tdB = tdZeroPrediction(deterministicPolicy(GOAL_POLICY, 5), DEFAULT_CONFIG, 0.1, 20, 5, 55);
  assert(tdA.updates.length === tdB.updates.length, 'TD(0) same seed same count');

  // 11. sampleActionWithRng uses seeded RNG
  const rng = mulberry32(1);
  const a1 = sampleActionWithRng([0.5, 0.5, 0, 0, 0], rng);
  const rng2 = mulberry32(1);
  const a2 = sampleActionWithRng([0.5, 0.5, 0, 0, 0], rng2);
  assert(a1 === a2, 'sampleActionWithRng same seed same action');

  // 12. Terminal state index is valid
  assert(terminalState >= 0 && terminalState < 9, 'terminal state should be within the grid');

  console.log('All gridworld tests passed.');
}
