import {
  DEFAULT_CONFIG,
} from './gridworld';
import {
  softmaxPolicy,
  stateFeatures,
  policyPreferences,
  softmaxScoreGradient,
  expectedScoreZero,
  expectedScoreZeroFeature,
  computeDiscountedReturns,
  reinforceBandit,
  reinforceWithBaseline,
  reinforceMDP,
  computePolicyMetrics,
  compareBaselineVariance,
} from './policyGradient';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(`Test failed: ${msg}`);
}

function near(a: number, b: number, eps = 1e-9) {
  return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= eps;
}

function arraysNear(a: number[], b: number[], eps = 1e-9) {
  if (a.length !== b.length) return false;
  return a.every((v, i) => near(v, b[i], eps));
}

function matricesNear(a: number[][], b: number[][], eps = 1e-9) {
  if (a.length !== b.length) return false;
  return a.every((row, i) => arraysNear(row, b[i], eps));
}

function sum(arr: number[]) {
  return arr.reduce((s, v) => s + v, 0);
}

export function runPGTests() {
  // 1. softmax probabilities sum to 1
  const prefs = [1, 2, 0.5, -1, 0];
  const policy = softmaxPolicy(prefs);
  assert(near(sum(policy), 1, 1e-9), 'softmax probabilities sum to 1');
  assert(policy.every((p) => p > 0 && p < 1), 'softmax probabilities are strictly positive');

  // 2. softmax shift invariance
  const shifted = softmaxPolicy(prefs.map((p) => p + 3));
  assert(arraysNear(policy, shifted, 1e-9), 'softmax is shift invariant');

  // 3. score function matches analytic formula
  const p = softmaxPolicy([0.5, -0.5, 0, 0.2, -0.2]);
  const action = 1;
  const score = softmaxScoreGradient(p, action);
  for (let a = 0; a < p.length; a++) {
    const expected = a === action ? 1 - p[a] : -p[a];
    assert(near(score[a], expected, 1e-12), `score mismatch at action ${a}`);
  }

  // 4. expected score is zero (tabular)
  const uniform = new Array(5).fill(1 / 5);
  const expZero = expectedScoreZero(uniform);
  assert(expZero.every((v) => near(v, 0, 1e-12)), 'expected score is zero for uniform policy');

  const skewed = softmaxPolicy([1, 3, 0, -0.5, 2]);
  const expZeroSkew = expectedScoreZero(skewed);
  assert(expZeroSkew.every((v) => near(v, 0, 1e-12)), 'expected score is zero for skewed policy');

  // 5. feature-based expected score is zero
  const coordPhi = stateFeatures(3, 'coordinate', DEFAULT_CONFIG);
  const coordPolicy = softmaxPolicy(policyPreferences(
    Array.from({ length: 5 }, () => new Array(coordPhi.length).fill(0)),
    3,
    'coordinate',
    DEFAULT_CONFIG
  ));
  const expZeroFeat = expectedScoreZeroFeature(coordPolicy, coordPhi);
  assert(expZeroFeat.every((v) => near(v, 0, 1e-12)), 'expected feature score is zero');

  // 6. discounted returns (Algorithm 9.1)
  const rewards = [1, 2, 3];
  const returns = computeDiscountedReturns(rewards, 0.9);
  assert(near(returns[0], 1 + 0.9 * (2 + 0.9 * 3), 1e-9), 'G_0 computed correctly');
  assert(near(returns[1], 2 + 0.9 * 3, 1e-9), 'G_1 computed correctly');
  assert(near(returns[2], 3, 1e-9), 'G_2 computed correctly');

  // 7. REINFORCE bandit reproducibility and convergence direction
  const actionRewards = [1, 3];
  const initTheta = [0, 0];
  const banditA = reinforceBandit(actionRewards, initTheta, 0.2, 300, 7, 0);
  const banditB = reinforceBandit(actionRewards, initTheta, 0.2, 300, 7, 7);
  const banditC = reinforceBandit(actionRewards, initTheta, 0.2, 300, 7, 0);
  assert(banditA.policyHistory.length === banditB.policyHistory.length, 'same episodes => same length');
  assert(
    banditA.policyHistory.every((p, i) => arraysNear(p, banditC.policyHistory[i], 1e-9)),
    'same seed produces identical policy history'
  );
  assert(
    banditA.records.some((_, i) => !arraysNear(banditA.policyHistory[i], banditB.policyHistory[i], 1e-9)),
    'different seeds produce different histories'
  );
  const finalPolicy = banditA.policyHistory[banditA.policyHistory.length - 1];
  assert(finalPolicy[1] > finalPolicy[0], 'bandit policy favors higher reward action');

  // 8. baseline uses baselineBefore and updates correctly
  const baselineResult = reinforceWithBaseline(actionRewards, initTheta, 0.2, 0.1, 300, 11, 0);
  baselineResult.records.forEach((rec) => {
    assert(rec.baselineBefore !== undefined, 'baselineBefore recorded');
    assert(rec.advantage !== undefined, 'advantage recorded');
    assert(near(rec.advantage!, rec.reward - rec.baselineBefore!, 1e-9), 'advantage uses baselineBefore');
  });
  let baseline = 0;
  baselineResult.records.forEach((rec) => {
    baseline += 0.1 * (rec.reward - baseline);
    assert(near(baseline, rec.baselineAfter!, 1e-9), 'baseline updates after actor update');
  });

  // 9. REINFORCE MDP reproducibility and behavior/update probabilities
  const mdpA = reinforceMDP(DEFAULT_CONFIG, {
    alpha: 0.05,
    episodes: 30,
    maxSteps: 20,
    seed: 42,
    useBaseline: false,
    featureMode: 'coordinate',
  });
  const mdpB = reinforceMDP(DEFAULT_CONFIG, {
    alpha: 0.05,
    episodes: 30,
    maxSteps: 20,
    seed: 42,
    useBaseline: false,
    featureMode: 'coordinate',
  });
  assert(mdpA.updateRecords.length === mdpB.updateRecords.length, 'update records length equal for same seed');
  mdpA.updateRecords.forEach((rec, i) => {
    const other = mdpB.updateRecords[i];
    assert(rec.episode === other.episode && rec.time === other.time, 'record indices match');
    assert(near(rec.returnGt, other.returnGt, 1e-9), `return mismatch at record ${i}`);
    assert(near(rec.behaviorProb, other.behaviorProb, 1e-9), `behaviorProb mismatch at record ${i}`);
    assert(near(rec.updateProb, other.updateProb, 1e-9), `updateProb mismatch at record ${i}`);
    assert(matricesNear(rec.parameterDelta, other.parameterDelta, 1e-9), `parameterDelta mismatch at record ${i}`);
  });

  // 10. behaviorProb matches behavior policy, updateProb matches update policy
  mdpA.updateRecords.forEach((rec) => {
    assert(near(rec.behaviorProb, rec.behaviorPolicy[rec.state][rec.action], 1e-9), 'behaviorProb matches behaviorPolicy');
    assert(near(rec.updateProb, rec.updatePolicy[rec.state][rec.action], 1e-9), 'updateProb matches updatePolicy');
  });

  // 11. G_t matches manual discounted-return formula; cumulativeReward and G0 separated
  mdpA.episodes.forEach((ep) => {
    const manual = computeDiscountedReturns(ep.trajectory.map((tr) => tr.reward), DEFAULT_CONFIG.gamma);
    assert(near(ep.discountedReturnG0, manual[0] ?? 0, 1e-9), 'discountedReturnG0 matches manual');
    ep.stepDetails.forEach((det, t) => {
      assert(near(det.returnGt, manual[t], 1e-9), 'per-step G_t matches manual');
    });
    const manualCumulative = ep.trajectory.reduce((s, tr) => s + tr.reward, 0);
    assert(near(ep.cumulativeReward, manualCumulative, 1e-9), 'cumulativeReward matches sum of rewards');
  });

  // 12. transition count and terminal/truncated flags
  mdpA.episodes.forEach((ep) => {
    assert(ep.episodeLength <= 20, 'episode length does not exceed H');
    assert(ep.episodeLength === ep.trajectory.length, 'episodeLength equals trajectory length');
    assert(ep.success === (ep.trajectory.length > 0 && ep.trajectory[ep.trajectory.length - 1].done), 'success flag consistent');
    assert(ep.truncated === (!ep.success && ep.trajectory.length === 20), 'truncated flag consistent');
  });

  // 13. policy metrics
  const numStates = DEFAULT_CONFIG.rows * DEFAULT_CONFIG.cols;
  const flatTheta = new Array(5).fill(0);
  const theta = Array.from({ length: 5 }, (_, a) =>
    flatTheta.map((_, i) => (i === a ? 0.5 : 0))
  );
  const d0 = new Array(numStates).fill(1 / numStates);
  const metrics = computePolicyMetrics(theta, DEFAULT_CONFIG, d0, 'onehot');
  assert(near(metrics.Jv0, sum(metrics.vPi.map((v, s) => d0[s] * v)), 1e-9), 'Jv0 matches d0-weighted value');
  assert(near(metrics.Jv, sum(metrics.vPi.map((v, s) => metrics.dPi[s] * v)), 1e-9), 'Jv matches dPi-weighted value');
  assert(near(metrics.Jr, sum(metrics.rPi.map((r, s) => metrics.dPi[s] * r)), 1e-9), 'Jr matches dPi-weighted reward');
  assert(near(sum(metrics.dPi), 1, 1e-6), 'stationary distribution sums to 1');
  assert(metrics.policy.every((dist) => near(sum(dist), 1, 1e-9)), 'each policy state distribution sums to 1');

  // 14. baseline variance comparison: means close, variances nonnegative
  const theta5 = [0.1, 0.5, -0.2, 0, 0];
  const rewards5 = [1, 3, 2, 1.5, 0.5];
  const varianceResult = compareBaselineVariance(theta5, rewards5, 1.5, 100);
  varianceResult.meanNoBaseline.forEach((m, i) => {
    assert(near(m, varianceResult.meanBaseline[i], 0.5), `means close for component ${i}`);
  });
  assert(varianceResult.varNoBaseline.every((v) => v >= 0), 'no-baseline variances nonnegative');
  assert(varianceResult.varBaseline.every((v) => v >= 0), 'baseline variances nonnegative');

  console.log('✅ Policy gradient tests passed');
}
