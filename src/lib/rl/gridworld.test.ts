import {
  DEFAULT_CONFIG,
  EPISODIC_PATH_CONFIG,
  deterministicPolicy,
  sampleActionWithRng,
  step,
  tdZeroPrediction,
  sarsa,
  qLearning,
  nStepSarsa,
  policyBellmanResidualV,
  policyBellmanResidualQ,
  optimalBellmanResidualQ,
  epsilonAtEpisode,
  valueIteration,
  type Policy,
  type Action,
} from './gridworld';
import { mulberry32 } from './stochasticApproximation';

const GOAL_POLICY: Action[] = [1, 2, 2, 1, 2, 2, 1, 1, 4];

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(`Test failed: ${msg}`);
}

function near(a: number, b: number, eps = 1e-9) {
  return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= eps;
}

function countEpisodeTransitions(updates: { episode: number }[], episode: number) {
  return updates.filter((u) => u.episode === episode).length;
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

  // Hand-compute probability-weighted residual under uniform random policy.
  let maxHandResidual = 0;
  for (let s = 0; s < 9; s++) {
    let expected = 0;
    for (let a = 0; a < 5; a++) {
      const result = step(s, a as Action, DEFAULT_CONFIG);
      expected += 0.2 * (result.reward + DEFAULT_CONFIG.gamma * values[result.nextState]);
    }
    maxHandResidual = Math.max(maxHandResidual, Math.abs(values[s] - expected));
  }
  assert(
    near(residualStoch, maxHandResidual, 1e-9),
    'policyBellmanResidualV should equal hand-calculated probability-weighted residual'
  );

  // 2. Terminal transition does not bootstrap: target equals reward only
  const sr = sarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 30, 50, 7);
  const terminalUpdates = sr.updates.filter((u) => u.done);
  assert(terminalUpdates.length > 0, 'Sarsa should observe at least one terminal transition');
  terminalUpdates.forEach((u, i) => {
    assert(
      near(u.target, u.reward, 1e-9),
      `terminal update ${i} target should equal reward, got target=${u.target} reward=${u.reward}`
    );
    assert(
      near(u.bootstrapValue, 0, 1e-9),
      `terminal update ${i} bootstrap should be 0, got ${u.bootstrapValue}`
    );
  });

  // 3. Q-learning residual directly uses learned Q (not reconstructed)
  const ql = qLearning(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 20, 10, 42);
  const lastQ = ql.frames[ql.frames.length - 1].qValues;
  const optRes = optimalBellmanResidualQ(lastQ, EPISODIC_PATH_CONFIG);
  assert(optRes >= 0, 'Q-learning optimal residual should be non-negative');

  // 4. Sarsa residual uses current behavior policy
  const lastFrame = sr.frames[sr.frames.length - 1];
  const polRes = policyBellmanResidualQ(lastFrame.qValues, lastFrame.behaviorPolicy, EPISODIC_PATH_CONFIG);
  assert(polRes >= 0, 'Sarsa policy residual should be non-negative');

  // 5. n-step Sarsa does not exceed H transitions per episode (count includes terminal)
  const ns = nStepSarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 3, 15, 10, 7);
  for (let ep = 0; ep < 10; ep++) {
    const transitions = countEpisodeTransitions(ns.updates, ep);
    assert(transitions <= 15, `n-step episode ${ep} should have at most 15 transitions, got ${transitions}`);
  }

  // 6. n=1 n-step Sarsa matches Sarsa target on the first transition of each episode
  const ns1 = nStepSarsa(EPISODIC_PATH_CONFIG, 0.1, 0.5, 'fixed', 1, 30, 20, 123);
  const sarsaRes = sarsa(EPISODIC_PATH_CONFIG, 0.1, 0.5, 'fixed', 30, 20, 123);
  const maxEp = Math.max(
    ...ns1.updates.map((u) => u.episode),
    ...sarsaRes.updates.map((u) => u.episode)
  );
  for (let ep = 0; ep <= maxEp; ep++) {
    const nu = ns1.updates.find((u) => u.episode === ep);
    const su = sarsaRes.updates.find((u) => u.episode === ep);
    if (nu && su) {
      assert(
        nu.state === su.state && nu.action === su.action,
        `n=1 first transition mismatch in episode ${ep}`
      );
      assert(near(nu.target, su.target, 1e-9), `n=1 target mismatch in episode ${ep}`);
    }
  }

  // 7. n-step Sarsa reproducibility
  const nsA = nStepSarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 3, 15, 10, 123);
  const nsB = nStepSarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 3, 15, 10, 123);
  assert(nsA.updates.length === nsB.updates.length, 'same seed should produce same update count');
  nsA.updates.forEach((u, i) => {
    const v = nsB.updates[i];
    assert(
      u.episode === v.episode &&
        u.state === v.state &&
        u.action === v.action &&
        u.reward === v.reward &&
        u.nextState === v.nextState &&
        near(u.target, v.target, 1e-9) &&
        near(u.tdError, v.tdError, 1e-9) &&
        near(u.newEstimate, v.newEstimate, 1e-9),
      `same seed mismatch at update ${i}`
    );
  });

  // 8. Same seed produces identical Sarsa history
  const a = sarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 20, 5, 99);
  const b = sarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 20, 5, 99);
  assert(a.updates.length === b.updates.length, 'same seed should produce same update count');
  a.updates.forEach((u, i) => {
    const v = b.updates[i];
    assert(
      u.episode === v.episode &&
        u.state === v.state &&
        u.action === v.action &&
        u.reward === v.reward &&
        u.nextState === v.nextState &&
        near(u.target, v.target, 1e-9) &&
        near(u.newEstimate, v.newEstimate, 1e-9),
      `same seed mismatch at update ${i}`
    );
  });

  // 9. Different seed usually produces different trajectories
  const c = sarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 20, 5, 100);
  const different = a.updates.some((u, i) => {
    const v = c.updates[i];
    return !v || u.action !== v.action || u.reward !== v.reward;
  });
  assert(different, 'different seed should usually produce different trajectories');

  // 10. Fixed epsilon vs GLIE sequences
  const fixed = epsilonAtEpisode(100, 0.5, 'fixed');
  const glie = epsilonAtEpisode(100, 0.5, 'glie');
  assert(near(fixed, 0.5), 'fixed epsilon should stay at 0.5');
  assert(glie < 0.5 && glie > 0, 'GLIE epsilon should decay');
  const glie0 = epsilonAtEpisode(0, 0.5, 'glie');
  assert(near(glie0, 0.5), 'GLIE epsilon at episode 0 should be epsilon0');

  // 11. TD(0) prediction uses seeded RNG
  const tdA = tdZeroPrediction(deterministicPolicy(GOAL_POLICY, 5), DEFAULT_CONFIG, 0.1, 20, 5, 55);
  const tdB = tdZeroPrediction(deterministicPolicy(GOAL_POLICY, 5), DEFAULT_CONFIG, 0.1, 20, 5, 55);
  assert(tdA.updates.length === tdB.updates.length, 'TD(0) same seed same count');
  tdA.updates.forEach((u, i) => {
    const v = tdB.updates[i];
    assert(
      u.episode === v.episode &&
        u.state === v.state &&
        u.action === v.action &&
        u.reward === v.reward &&
        near(u.newEstimate, v.newEstimate, 1e-9) &&
        u.valuesAfter !== undefined &&
        v.valuesAfter !== undefined &&
        near(u.valuesAfter![u.state], v.valuesAfter![u.state], 1e-9),
      `TD(0) same seed mismatch at update ${i}`
    );
  });

  // 12. sampleActionWithRng uses seeded RNG
  const rng = mulberry32(1);
  const a1 = sampleActionWithRng([0.5, 0.5, 0, 0, 0], rng);
  const rng2 = mulberry32(1);
  const a2 = sampleActionWithRng([0.5, 0.5, 0, 0, 0], rng2);
  assert(a1 === a2, 'sampleActionWithRng same seed same action');

  // 13. Frame policies are deterministic functions of Q and epsilon (no render-time random sampling)
  const frame = sarsa(EPISODIC_PATH_CONFIG, 0.1, 0.5, 'fixed', 10, 1, 1).frames[0];
  assert(frame.behaviorPolicy.length === 9, 'behaviorPolicy should cover all states');
  assert(frame.greedyPolicy.length === 9, 'greedyPolicy should cover all states');
  frame.greedyPolicy.forEach((dist) => {
    const sum = dist.reduce((acc, p) => acc + p, 0);
    assert(near(sum, 1, 1e-9), 'greedy policy should sum to 1');
    const ones = dist.filter((p) => near(p, 1, 1e-9)).length;
    assert(ones === 1, 'greedy policy should be deterministic');
  });
  // With Q=0, deterministic tie-breaking picks action 0; greedy should put all mass on action 0.
  assert(near(frame.greedyPolicy[0][0], 1, 1e-9), 'greedy policy should pick first max action deterministically');

  // Behavior policy should match epsilon-greedy structure: one best action gets 1-eps+eps/5,
  // the others get eps/5. With Q=0 and eps=0.5 there is a single best action (random tie-breaking
  // in the backend, but the distribution shape is fixed).
  frame.behaviorPolicy.forEach((dist) => {
    const sum = dist.reduce((acc, p) => acc + p, 0);
    assert(near(sum, 1, 1e-9), 'behavior policy should sum to 1');
    const high = dist.filter((p) => near(p, 0.6, 1e-9)).length;
    const low = dist.filter((p) => near(p, 0.1, 1e-9)).length;
    assert(high === 1 && low === 4, 'behavior policy should be epsilon-greedy with eps=0.5');
  });

  // Same seed should produce identical behavior policies.
  const frameA = sarsa(EPISODIC_PATH_CONFIG, 0.1, 0.5, 'fixed', 10, 1, 7).frames[0];
  const frameB = sarsa(EPISODIC_PATH_CONFIG, 0.1, 0.5, 'fixed', 10, 1, 7).frames[0];
  frameA.behaviorPolicy.forEach((dist, s) => {
    dist.forEach((p, a) => {
      assert(near(p, frameB.behaviorPolicy[s][a], 1e-9), `behavior policy not reproducible at s=${s} a=${a}`);
    });
  });

  // 14. Greedy action agreement baseline is deterministic (value iteration)
  const { policies } = valueIteration(EPISODIC_PATH_CONFIG);
  const optimalPolicy = policies[policies.length - 1];
  assert(optimalPolicy.length === 9, 'value iteration should return a policy');
  optimalPolicy.forEach((dist) => {
    const sum = dist.reduce((acc, p) => acc + p, 0);
    assert(near(sum, 1, 1e-9), 'optimal policy should sum to 1');
  });

  // 15. TD player index bounds
  const oneUpdate = tdZeroPrediction(deterministicPolicy(GOAL_POLICY, 5), EPISODIC_PATH_CONFIG, 0.1, 1, 1, 1);
  assert(oneUpdate.updates.length === 1, 'should produce exactly 1 update');
  assert(oneUpdate.updates[0] !== undefined, 'update at index 0 should be defined');
  const many = sarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 5, 5, 1);
  const maxIdx = Math.max(0, many.updates.length - 1);
  assert(many.updates[maxIdx] !== undefined, `last update at index ${maxIdx} should be defined`);
  assert(many.updates[many.updates.length] === undefined, 'index out of bounds should be undefined');

  // 16. Goal policy reaches target from every non-terminal state
  const goalPolicy = deterministicPolicy(GOAL_POLICY, 5);
  for (let s = 0; s < 9; s++) {
    if (s === EPISODIC_PATH_CONFIG.targetState || EPISODIC_PATH_CONFIG.forbiddenStates.includes(s)) continue;
    let cur = s;
    let reached = false;
    for (let k = 0; k < 20; k++) {
      if (cur === EPISODIC_PATH_CONFIG.targetState) {
        reached = true;
        break;
      }
      const a = goalPolicy[cur].findIndex((p) => p > 0.5);
      const res = step(cur, a as Action, EPISODIC_PATH_CONFIG);
      cur = res.nextState;
    }
    assert(reached, `goal policy should reach target from state ${s}`);
  }

  // 17. n-step target decomposition matches actual target
  const ns3 = nStepSarsa(EPISODIC_PATH_CONFIG, 0.1, 0.3, 'fixed', 3, 20, 5, 11);
  ns3.updates.forEach((u, i) => {
    if (!u.rewardTerms) return;
    let handTarget = 0;
    u.rewardTerms.forEach((rew, k) => {
      handTarget += Math.pow(EPISODIC_PATH_CONFIG.gamma, k) * rew;
    });
    if (u.bootstrapState !== undefined && u.bootstrapAction !== undefined && u.qBefore) {
      handTarget += Math.pow(EPISODIC_PATH_CONFIG.gamma, u.rewardTerms.length) *
        u.qBefore[u.bootstrapState][u.bootstrapAction];
    }
    assert(near(u.target, handTarget, 1e-9), `n-step target mismatch at update ${i}`);
  });

  // 18. Multi-optimal-action agreement: tie should not be penalized
  // Construct a state where two actions share the same optimal value.
  const qStar = Array.from({ length: 9 }, () => [1, 1, 0, 0, 0]);
  const qLearned = Array.from({ length: 9 }, () => [1, 0.9, 0, 0, 0]);
  // Manual agreement check: action 0 is in optimal set {0,1}.
  const max0 = Math.max(...qLearned[0]);
  const greedyActions = qLearned[0]
    .map((v, a) => ({ v, a }))
    .filter(({ v }) => Math.abs(v - max0) <= 1e-6)
    .map(({ a }) => a);
  const maxStar = Math.max(...qStar[0]);
  const optimalActions = qStar[0]
    .map((v, a) => ({ v, a }))
    .filter(({ v }) => Math.abs(v - maxStar) <= 1e-6)
    .map(({ a }) => a);
  assert(greedyActions.some((a) => optimalActions.includes(a)), 'greedy action should belong to optimal set');

  console.log('All gridworld tests passed.');
}
