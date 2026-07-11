import {
  mulberry32,
  normalRandom,
  generateNormalSamples,
  batchVsIncrementalMean,
  fullObjective,
  meanEstimationGradientDescent,
  estimateBatchGradientVariance,
  tdBridgeStep,
  checkDvoretzkyAlphaPower,
  checkDvoretzkyBetaPower,
} from './stochasticApproximation';
import { deterministicPolicy, DEFAULT_CONFIG, type Action } from './gridworld';

const GOAL_POLICY: Action[] = [2, 2, 3, 1, 2, 3, 0, 1, 4];

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(`Test failed: ${msg}`);
}

function near(a: number, b: number, eps = 1e-9) {
  return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= eps;
}

export function runStochasticApproximationTests() {
  // 1. batch mean equals incremental mean at every step
  const samples = generateNormalSamples(50, 3, 1, 42);
  const meanHistory = batchVsIncrementalMean(samples);
  meanHistory.forEach((h) =>
    assert(near(h.batchMean, h.incrementalMean), `batch/inc mismatch at n=${h.n}`)
  );

  // 2. same seed gives identical samples
  const s1 = generateNormalSamples(20, 0, 1, 7);
  const s2 = generateNormalSamples(20, 0, 1, 7);
  s1.forEach((x, i) => assert(near(x, s2[i]), `seed mismatch at index ${i}`));

  // 3. fullLossAfter equals fullObjective(dataset, wAfter)
  const dataset = generateNormalSamples(30, 2, 0.5, 99);
  const bgdHistory = meanEstimationGradientDescent(dataset, 0, 'bgd', 30, 0.1, 5, 99);
  bgdHistory.forEach((step, idx) =>
    assert(
      near(step.fullLossAfter, fullObjective(dataset, step.wAfter)),
      `fullLossAfter mismatch at step ${idx}`
    )
  );

  // 4. BGD gradient noise is zero
  bgdHistory.forEach((step, idx) =>
    assert(near(step.gradientNoise, 0), `BGD gradientNoise not zero at step ${idx}`)
  );

  // 5. SGD squared gradient noise is not identically zero
  const sgdHistory = meanEstimationGradientDescent(dataset, 0, 'sgd', 1, 0.1, 5, 99);
  const sgdNonzero = sgdHistory.some((step) => step.squaredGradientNoise > 1e-12);
  assert(sgdNonzero, 'SGD squaredGradientNoise is identically zero');

  // Fixed-w gradient variance: SGD > mini-batch > 0
  const mbgdNoise = estimateBatchGradientVariance(dataset, 0, 'mbgd', 10, 200, 99);
  const sgdNoise = estimateBatchGradientVariance(dataset, 0, 'sgd', 1, 200, 99);
  assert(
    sgdNoise.varianceOfBatchGradients > mbgdNoise.varianceOfBatchGradients,
    'SGD variance should exceed mini-batch variance'
  );

  // 6. TD next state equals previous nextState (continuous trajectory)
  const policy = deterministicPolicy(GOAL_POLICY, 5);
  const v0 = new Array(9).fill(0);
  const rng = mulberry32(123);
  const r1 = tdBridgeStep(DEFAULT_CONFIG, policy, v0, 0.2, DEFAULT_CONFIG.startState, rng);
  assert(
    r1.nextCurrentState === r1.step.nextState,
    'TD nextCurrentState should equal step.nextState'
  );
  const r2 = tdBridgeStep(
    DEFAULT_CONFIG,
    policy,
    r1.vNew,
    0.2,
    r1.nextCurrentState,
    rng
  );
  assert(
    r2.step.state === r1.nextCurrentState,
    'TD continuity broken: r2.state != r1.nextState'
  );

  // 7. Terminal transition does not bootstrap
  const episodicConfig = { ...DEFAULT_CONFIG, taskType: 'episodic' as const };
  const rng2 = mulberry32(1);
  // From state 7 the goal policy goes right to target state 8 (done=true in episodic)
  const rt = tdBridgeStep(
    episodicConfig,
    policy,
    new Array(9).fill(5),
    0.2,
    7,
    rng2
  );
  assert(rt.step.done, 'terminal transition should be done');
  assert(near(rt.step.tdTarget, rt.step.reward), 'terminal tdTarget should equal reward');
  assert(rt.step.vNextOld === 0, 'terminal vNextOld should be 0');
  assert(rt.nextCurrentState === episodicConfig.startState, 'terminal should reset to startState');

  // 8. normalRandom always returns finite values
  const rng3 = mulberry32(0);
  for (let i = 0; i < 1000; i++) {
    const z = normalRandom(rng3, 0, 1);
    assert(Number.isFinite(z), 'normalRandom produced non-finite value');
  }

  // 9. Dvoretzky alpha/beta power conditions
  assert(checkDvoretzkyAlphaPower(0.75).valid, 'pAlpha=0.75 should be valid');
  assert(!checkDvoretzkyAlphaPower(1.25).valid, 'pAlpha=1.25 should be invalid');
  assert(checkDvoretzkyBetaPower(0.75).valid, 'pBeta=0.75 should be valid');
  assert(checkDvoretzkyBetaPower(1.25).valid, 'pBeta=1.25 should be valid');
  assert(!checkDvoretzkyBetaPower(0.5).valid, 'pBeta=0.5 should be invalid');

  console.log('All stochastic approximation tests passed.');
}
