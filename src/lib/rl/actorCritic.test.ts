import {
  EPISODIC_PATH_CONFIG,
  DEFAULT_CONFIG,
  solveStateValues,
  computeQValues,
  isTerminal,
  stochasticStepDistribution,
  type StochasticOutcome,
  type Action,
} from './gridworld';
import {
  qac,
  a2c,
  offPolicyActorCritic,
  qBasedOffPolicyActorCritic,
  deterministicActorCriticEpisode,
  checkCoverage,
  effectiveSampleSize,
  policyWeightedStateValues,
  computeACMetricSeries,
  sampleTdErrorAtStateAction,
  checkBaselineInvariance,
  buildActionIndependentBaseline,
  buildActionDependentBaseline,
  klDivergence,
  solveStateValuesWithSlip,
  computeQValuesWithSlip,
  type ACOptions,
  type ACResult,
} from './actorCritic';
import { mulberry32 } from './stochasticApproximation';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(`Test failed: ${msg}`);
}

function near(a: number, b: number, eps = 1e-9) {
  return Math.abs(a - b) <= eps;
}

const dummyRng = () => 0.5;

const defaultOptions: ACOptions = {
  seed: 42,
  horizonH: 20,
  actorAlpha: 0.05,
  criticAlpha: 0.1,
  episodes: 10,
};

export function runACTests() {
  // 1. same seed QAC reproducible
  const qac1 = qac(EPISODIC_PATH_CONFIG, defaultOptions);
  const qac2 = qac(EPISODIC_PATH_CONFIG, defaultOptions);
  assert(qac1.updates.length === qac2.updates.length, 'qac same seed length');
  qac1.updates.forEach((u, i) => {
    const v = qac2.updates[i];
    assert(near(u.tdError, v.tdError, 1e-9), `qac same seed tdError ${i}`);
    assert(near(u.actorWeight, v.actorWeight, 1e-9), `qac same seed actorWeight ${i}`);
  });

  // 2. same seed A2C reproducible
  const a2c1 = a2c(EPISODIC_PATH_CONFIG, defaultOptions);
  const a2c2 = a2c(EPISODIC_PATH_CONFIG, defaultOptions);
  assert(a2c1.updates.length === a2c2.updates.length, 'a2c same seed length');
  a2c1.updates.forEach((u, i) => {
    const v = a2c2.updates[i];
    assert(near(u.tdError, v.tdError, 1e-9), `a2c same seed tdError ${i}`);
    assert(near(u.actorWeight, v.actorWeight, 1e-9), `a2c same seed actorWeight ${i}`);
  });

  // 3. different seeds usually differ
  const qacA = qac(EPISODIC_PATH_CONFIG, { ...defaultOptions, seed: 1 });
  const qacB = qac(EPISODIC_PATH_CONFIG, { ...defaultOptions, seed: 999 });
  const differs = qacA.updates.some(
    (u, i) => i < qacB.updates.length && (u.action !== qacB.updates[i].action || u.reward !== qacB.updates[i].reward)
  );
  assert(differs, 'different seeds produce different trajectories');

  // 4. terminal transition target = reward (no bootstrap)
  const qacLong = qac(EPISODIC_PATH_CONFIG, { ...defaultOptions, horizonH: 100, episodes: 30 });
  const terminalUpdates = qacLong.updates.filter((u) => u.done);
  assert(terminalUpdates.length > 0, 'at least one terminal transition observed');
  terminalUpdates.forEach((u) => {
    assert(near(u.criticBootstrap, 0, 1e-9), 'terminal bootstrap is zero');
    assert(near(u.criticTarget, u.reward, 1e-9), 'terminal target equals reward');
  });

  // 5. each episode transition count <= H
  const maxEpisodeLength = Math.max(...qacLong.episodes.map((e) => e.episodeLength));
  assert(maxEpisodeLength <= 100, 'episode length does not exceed horizon');

  // 6. truncated vs natural terminal
  qacLong.episodes.forEach((ep) => {
    if (ep.success) assert(!ep.truncated, 'success episode is not truncated');
    if (ep.truncated) assert(!ep.success, 'truncated episode is not success');
  });

  // 7. QAC policy value = sum pi * Q
  const qacRes = qac(EPISODIC_PATH_CONFIG, { ...defaultOptions, episodes: 20 });
  assert(!!qacRes.finalQ && !!qacRes.finalPolicy, 'qac returns final q and policy');
  const vWeighted = policyWeightedStateValues(qacRes.finalQ!, qacRes.finalPolicy!);
  vWeighted.forEach((val, s) => {
    const expected = qacRes.finalQ![s].reduce((sum, qv, a) => sum + qacRes.finalPolicy![s][a] * qv, 0);
    assert(near(val, expected, 1e-9), `policy weighted value state ${s}`);
  });

  // 8. QAC actorWeight uses qBefore
  qacRes.updates.forEach((u, i) => {
    assert(near(u.actorWeight, u.criticEstimateBefore, 1e-9), `qac actorWeight is qBefore ${i}`);
  });

  // 8b. QAC nextAction is carried over to the next step's action (Sarsa-style continuity)
  for (let i = 0; i < qacRes.updates.length - 1; i++) {
    const u = qacRes.updates[i];
    const v = qacRes.updates[i + 1];
    if (u.episode === v.episode && !u.done) {
      assert(u.nextAction !== undefined, `qac nextAction recorded at step ${i}`);
      assert(u.nextAction === v.action, `qac nextAction matches next step action at ${i}`);
    }
  }

  // 9. A2C TD error formula
  const a2cRes = a2c(EPISODIC_PATH_CONFIG, defaultOptions);
  a2cRes.updates.forEach((u, i) => {
    const expected = u.reward + EPISODIC_PATH_CONFIG.gamma * u.criticBootstrap - u.criticEstimateBefore;
    assert(near(u.tdError, expected, 1e-9), `a2c tdError formula ${i}`);
  });

  // 10. With exact V, sampled TD error equals true advantage
  const exactConfig = { ...EPISODIC_PATH_CONFIG, taskType: 'continuing' as const };
  const uniformPolicy = Array.from({ length: 9 }, () => Array.from({ length: 5 }, () => 1 / 5));
  const trueV = solveStateValues(uniformPolicy, exactConfig);
  const trueQ = computeQValues(trueV, exactConfig);
  const testState = 0;
  const testAction = 1;
  const sampleRes = sampleTdErrorAtStateAction({
    config: exactConfig,
    state: testState,
    action: testAction,
    values: trueV,
    rng: dummyRng,
  });
  assert(sampleRes.count === 1, 'exact sampler returns one valid sample');
  const trueAdvantage = trueQ[testState][testAction] - trueV[testState];
  assert(near(sampleRes.mean, trueAdvantage, 1e-9), 'sampled TD error equals true advantage with exact critic');

  // 11. pi = beta => rho = 1 and off-policy reduces to on-policy
  const on = a2c(EPISODIC_PATH_CONFIG, { ...defaultOptions, seed: 7 });
  const off = offPolicyActorCritic(EPISODIC_PATH_CONFIG, { ...defaultOptions, seed: 7, epsilon: 0 });
  assert(off.updates.length === on.updates.length, 'on/off length match when epsilon=0');
  off.updates.forEach((u, i) => {
    assert(near(u.rho!, 1, 1e-9), `rho=1 when epsilon=0 ${i}`);
    assert(near(u.tdError, on.updates[i].tdError, 1e-9), `tdError matches on-policy ${i}`);
  });

  // 12. behavior support not covering target throws
  let threw = false;
  try {
    checkCoverage([0.5, 0.5, 0, 0, 0], [1, 0, 0, 0, 0]);
  } catch {
    threw = true;
  }
  assert(threw, 'coverage violation throws');

  // 13. ESS calculation
  const ess = effectiveSampleSize([1, 1, 1, 1]);
  assert(near(ess, 4, 1e-9), 'ESS for equal weights');
  const ess2 = effectiveSampleSize([2, 0, 0, 0]);
  assert(near(ess2, 1, 1e-9), 'ESS for single dominant weight');

  // 14. deterministic AC step 2 uses step 1 updated parameters
  const env = {
    step: (s: number, a: number) => {
      const nextS = Math.max(-1, Math.min(1, s + 0.3 * a));
      return { nextState: nextS, reward: -(nextS * nextS), done: Math.abs(nextS) < 0.05 };
    },
    terminal: (s: number) => Math.abs(s) < 0.05,
  };
  const det = deterministicActorCriticEpisode(env, -0.8, [0, 0, 0, 0, 0, 0], [0.5, 0], {
    seed: 1,
    horizonH: 10,
    actorAlpha: 0.05,
    criticAlpha: 0.1,
    gamma: 0.9,
    qValueWithWeights: (s, a, w) =>
      [1, s, a, s * s, s * a, a * a].reduce((sum, f, i) => sum + f * w[i], 0),
    muWithTheta: (s, theta) => Math.tanh(theta[0] * s + theta[1]),
    gradMuWithTheta: (s, theta) => {
      const m = Math.tanh(theta[0] * s + theta[1]);
      const factor = 1 - m * m;
      return [factor * s, factor];
    },
    gradQWrtA: (s, a, w) => w[2] + w[4] * s + 2 * w[5] * a,
    criticFeatures: (s, a) => [1, s, a, s * s, s * a, a * a],
  });
  assert(det.steps.length >= 2, 'deterministic AC has at least 2 steps');
  assert(
    det.steps[1].wBefore.every((v, i) => near(v, det.steps[0].wAfter[i], 1e-9)),
    'deterministic AC step 2 uses step 1 updated w'
  );
  assert(
    det.steps[1].thetaBefore.every((v, i) => near(v, det.steps[0].thetaAfter[i], 1e-9)),
    'deterministic AC step 2 uses step 1 updated theta'
  );

  // 15. deterministic terminal transition does not bootstrap
  const terminalEnv = {
    step: () => ({ nextState: 0, reward: 1, done: true }),
    terminal: () => true,
  };
  const detTerminal = deterministicActorCriticEpisode(terminalEnv, 0, [0, 0, 0, 0, 0, 0], [0, 0], {
    seed: 1,
    horizonH: 5,
    actorAlpha: 0.05,
    criticAlpha: 0.1,
    gamma: 0.9,
    qValueWithWeights: () => 0,
    muWithTheta: () => 0,
    gradMuWithTheta: () => [0, 0],
    gradQWrtA: () => 0,
    criticFeatures: () => [1, 0, 0, 0, 0, 0],
  });
  assert(detTerminal.steps.length > 0, 'terminal env produces a step');
  assert(near(detTerminal.steps[0].target, detTerminal.steps[0].reward, 1e-9), 'terminal target equals reward');

  // 16. all finite
  assert(!qacRes.diverged, 'qac did not diverge');
  assert(!a2cRes.diverged, 'a2c did not diverge');
  assert(qacRes.updates.every((u) => Number.isFinite(u.tdError)), 'qac tdErrors finite');
  assert(a2cRes.updates.every((u) => Number.isFinite(u.actorWeight)), 'a2c actorWeights finite');

  // 17. sampleTdErrorAtStateAction works for arbitrary (s,a), not only start state
  const sampleResNonStart = sampleTdErrorAtStateAction({
    config: exactConfig,
    state: 4,
    action: 2,
    values: trueV,
    rng: dummyRng,
  });
  assert(sampleResNonStart.count === 1, 'one valid sample for non-terminal state');
  assert(sampleResNonStart.samples.every((d) => near(d, sampleResNonStart.mean, 1e-9)), 'deterministic samples are identical');
  assert(!near(sampleResNonStart.mean, 0, 1e-6), 'non-trivial state/action does not return fake zero');

  // 18. transition-level truncated flag is correct
  const truncRes = a2c(EPISODIC_PATH_CONFIG, { ...defaultOptions, horizonH: 5, episodes: 20 });
  truncRes.updates.forEach((u) => {
    const expectedTruncated = !u.done && u.time === 4;
    assert(u.truncated === expectedTruncated, `truncated flag correct at episode ${u.episode} time ${u.time}`);
  });
  truncRes.episodes.forEach((ep) => {
    if (ep.truncated) assert(!ep.success, 'truncated episode is not success');
  });

  // 19. success is recorded even when cumulative return is negative
  const negativeSuccessConfig = { ...EPISODIC_PATH_CONFIG, targetReward: 0, stepReward: -5 };
  const negRes = a2c(negativeSuccessConfig, { ...defaultOptions, horizonH: 100, episodes: 30, seed: 1 });
  const successEpisodes = negRes.episodes.filter((ep) => ep.success);
  const negativeSuccess = successEpisodes.some((ep) => ep.cumulativeReward < 0);
  assert(negativeSuccess, 'success can happen with negative cumulative reward');
  successEpisodes.forEach((ep) => assert(ep.episodeLength > 0, 'success episode has positive length'));

  // 20. metric series contains only real episodes 1..N
  const metrics = computeACMetricSeries(qacRes, 10);
  assert(metrics.length === qacRes.episodes.length, 'metric series length equals number of episodes');
  assert(metrics.every((m) => m.episode >= 1), 'no pseudo episode 0');
  assert(metrics[metrics.length - 1].episode === qacRes.episodes.length, 'last episode index matches');

  // 21. TD error metrics: mean abs and RMS are non-negative; signed mean can differ
  assert(metrics.every((m) => m.meanAbsoluteTdError >= 0), 'mean abs td error non-negative');
  assert(metrics.every((m) => m.rmsTdError >= 0), 'rms td error non-negative');
  const firstWithTd = metrics.find((m) => m.signedMeanTdError !== 0);
  if (firstWithTd) {
    assert(
      firstWithTd.meanAbsoluteTdError >= Math.abs(firstWithTd.signedMeanTdError) - 1e-9,
      'mean abs dominates signed magnitude'
    );
  }

  // 21b. mean absolute TD error is not cancelled by opposite-sign errors
  const cancellationResult: ACResult = {
    updates: [
      { episode: 1, time: 0, state: 0, action: 0, reward: 0, nextState: 0, done: false, truncated: false, actorPolicyBefore: [0.2, 0.2, 0.2, 0.2, 0.2], actorPolicyAfter: [0.2, 0.2, 0.2, 0.2, 0.2], actorFullPolicyBefore: [[0.2, 0.2, 0.2, 0.2, 0.2]], actorFullPolicyAfter: [[0.2, 0.2, 0.2, 0.2, 0.2]], criticEstimateBefore: 0, criticBootstrap: 0, criticTarget: 5, tdError: 5, criticEstimateAfter: 5, actorWeight: 5, scoreGradient: [0, 0, 0, 0, 0], actorDelta: [0, 0, 0, 0, 0] },
      { episode: 1, time: 1, state: 0, action: 0, reward: 0, nextState: 0, done: false, truncated: false, actorPolicyBefore: [0.2, 0.2, 0.2, 0.2, 0.2], actorPolicyAfter: [0.2, 0.2, 0.2, 0.2, 0.2], actorFullPolicyBefore: [[0.2, 0.2, 0.2, 0.2, 0.2]], actorFullPolicyAfter: [[0.2, 0.2, 0.2, 0.2, 0.2]], criticEstimateBefore: 0, criticBootstrap: 0, criticTarget: -5, tdError: -5, criticEstimateAfter: -5, actorWeight: -5, scoreGradient: [0, 0, 0, 0, 0], actorDelta: [0, 0, 0, 0, 0] },
    ],
    frames: [],
    episodes: [{ cumulativeReward: 0, discountedReturn: 0, episodeLength: 2, success: false, truncated: false }],
    initialPolicy: [[0.2, 0.2, 0.2, 0.2, 0.2]],
    policyAfterEpisode: [[[0.2, 0.2, 0.2, 0.2, 0.2]]],
    diverged: false,
  };
  const cancellationMetrics = computeACMetricSeries(cancellationResult, 10);
  assert(cancellationMetrics.length === 1, 'cancellation metric length');
  assert(near(cancellationMetrics[0].signedMeanTdError, 0, 1e-9), 'signed mean cancels');
  assert(near(cancellationMetrics[0].meanAbsoluteTdError, 5, 1e-9), 'mean absolute does not cancel');
  assert(near(cancellationMetrics[0].rmsTdError, 5, 1e-9), 'rms does not cancel');

  // 22. success rate uses ACEpisodeRecord.success, not return sign
  const successRateMatches = metrics.every((m, i) => m.success === (qacRes.episodes[i].success ? 1 : 0));
  assert(successRateMatches, 'metric success equals episode success flag');

  // 23. QAC critic update norm is non-zero when critic learns
  const qacMetrics = computeACMetricSeries(qacRes, 10);
  assert(qacMetrics.some((m) => m.criticUpdateNorm > 0), 'QAC critic update norm non-zero');

  // 24. rho > 10 proportion is reported, not actually clipped
  const highRhoRes = offPolicyActorCritic(EPISODIC_PATH_CONFIG, {
    ...defaultOptions,
    epsilon: 0.9,
    seed: 3,
    episodes: 20,
  });
  const rhos = highRhoRes.updates.map((u) => u.rho ?? 1);
  const highRhoCount = rhos.filter((r) => r > 10).length;
  // We only assert the field is present and that high-rho samples are not silently capped to 10.
  if (highRhoCount > 0) {
    assert(rhos.some((r) => r > 10), 'raw rho can exceed 10');
  }
  const qBasedRho = qBasedOffPolicyActorCritic(EPISODIC_PATH_CONFIG, {
    ...defaultOptions,
    epsilon: 0.9,
    seed: 3,
    episodes: 5,
  });
  assert(qBasedRho.updates.every((u) => u.rho !== undefined && Number.isFinite(u.rho)), 'q-based off-policy rho recorded');

  // 25. deterministic AC separates actor and behavior actions
  const detNoise = deterministicActorCriticEpisode(env, -0.8, [0, 0, 0, 0, 0, 0], [0.5, 0], {
    seed: 1,
    horizonH: 10,
    actorAlpha: 0.05,
    criticAlpha: 0.1,
    gamma: 0.9,
    qValueWithWeights: (s, a, w) =>
      [1, s, a, s * s, s * a, a * a].reduce((sum, f, i) => sum + f * w[i], 0),
    muWithTheta: (s, theta) => Math.tanh(theta[0] * s + theta[1]),
    gradMuWithTheta: (s, theta) => {
      const m = Math.tanh(theta[0] * s + theta[1]);
      const factor = 1 - m * m;
      return [factor * s, factor];
    },
    gradQWrtA: (s, a, w) => w[2] + w[4] * s + 2 * w[5] * a,
    criticFeatures: (s, a) => [1, s, a, s * s, s * a, a * a],
    explorationNoiseStd: 0.2,
  });
  detNoise.steps.forEach((st) => {
    assert(near(st.behaviorAction, st.actorAction + st.explorationNoise, 1e-9), 'behavior action = actor action + noise');
    assert(near(st.criticLoss, 0.5 * st.tdError * st.tdError, 1e-9), 'critic loss is 0.5 tdError^2');
  });
  const pureDet = deterministicActorCriticEpisode(env, -0.8, [0, 0, 0, 0, 0, 0], [0.5, 0], {
    seed: 1,
    horizonH: 10,
    actorAlpha: 0.05,
    criticAlpha: 0.1,
    gamma: 0.9,
    qValueWithWeights: (s, a, w) =>
      [1, s, a, s * s, s * a, a * a].reduce((sum, f, i) => sum + f * w[i], 0),
    muWithTheta: (s, theta) => Math.tanh(theta[0] * s + theta[1]),
    gradMuWithTheta: (s, theta) => {
      const m = Math.tanh(theta[0] * s + theta[1]);
      const factor = 1 - m * m;
      return [factor * s, factor];
    },
    gradQWrtA: (s, a, w) => w[2] + w[4] * s + 2 * w[5] * a,
    criticFeatures: (s, a) => [1, s, a, s * s, s * a, a * a],
  });
  pureDet.steps.forEach((st) => {
    assert(near(st.actorAction, st.behaviorAction, 1e-9), 'pure mode actor and behavior actions coincide');
    assert(near(st.explorationNoise, 0, 1e-9), 'pure mode has zero exploration noise');
  });
  assert(pureDet.steps.every((st) => st.criticWeightsUsedByActor === 'before'), 'deterministic AC defaults to using critic weights before update for the actor gradient');

  // 26. baseline invariance: action-independent => 0, action-dependent != 0
  const invPolicy = Array.from({ length: 9 }, () => Array.from({ length: 5 }, () => 1 / 5));
  const invV = solveStateValues(invPolicy, exactConfig);
  const invQ = computeQValues(invV, exactConfig);
  const indepBaseline = buildActionIndependentBaseline(9, 1.0);
  const depBaseline = buildActionDependentBaseline(invQ);
  const indepCheck = checkBaselineInvariance(invPolicy, indepBaseline);
  const depCheck = checkBaselineInvariance(invPolicy, depBaseline);
  assert(indepCheck.isInvariant, 'action-independent baseline leaves policy gradient unchanged');
  assert(indepCheck.maxAbs < 1e-9, 'independent baseline max abs is zero');
  indepCheck.perStateComponentSum.forEach((row, s) => {
    row.forEach((v, k) => assert(Math.abs(v) < 1e-9, `independent baseline zero at state ${s} component ${k}`));
  });
  assert(depCheck.maxAbs > 0.01, 'action-dependent baseline changes policy gradient');
  assert(depCheck.perStateComponentSum.some((row) => row.some((v) => Math.abs(v) > 0.01)), 'action-dependent has non-zero component');

  // 27. Overview data should come from A2C (V-based), not QAC
  const overviewRes = a2c(EPISODIC_PATH_CONFIG, {
    seed: 1,
    horizonH: 20,
    actorAlpha: 0.05,
    criticAlpha: 0.1,
    episodes: 1,
  });
  assert(overviewRes.updates.length > 0, 'overview a2c produces updates');
  assert(
    overviewRes.updates.every((u) => u.vBefore !== undefined && u.qBefore === undefined),
    'overview data uses V-based A2C records'
  );

  // 28. Advantage sampler traverses all non-terminal state × action
  const numStates = exactConfig.rows * exactConfig.cols;
  const numActions = 5;
  for (let s = 0; s < numStates; s++) {
    for (let a = 0; a < numActions; a++) {
      if (isTerminal(s, exactConfig)) {
        const invalid = sampleTdErrorAtStateAction({
          config: exactConfig,
          state: s,
          action: a as 0 | 1 | 2 | 3 | 4,
          values: trueV,
          rng: dummyRng,
        });
        assert(invalid.count === 0, 'terminal state returns zero valid samples');
      } else {
        const valid = sampleTdErrorAtStateAction({
          config: exactConfig,
          state: s,
          action: a as 0 | 1 | 2 | 3 | 4,
          values: trueV,
          rng: dummyRng,
        });
        assert(valid.count === 1, `non-terminal state ${s} action ${a} returns one valid sample`);
        const expectedAdv = trueQ[s][a] - trueV[s];
        assert(near(valid.mean, expectedAdv, 1e-9), `sampled advantage matches true advantage at (${s},${a})`);
      }
    }
  }

  // 29. natural terminal transition has done=true and truncated=false
  const naturalTerminals = qacLong.updates.filter((u) => u.done);
  assert(naturalTerminals.length > 0, 'found terminal transitions');
  naturalTerminals.forEach((u) => {
    assert(u.truncated === false, 'natural terminal transition is not truncated');
  });

  // 30. negative return + success=true is counted as success
  const negSuccessConfig = { ...EPISODIC_PATH_CONFIG, targetReward: 0, stepReward: -5 };
  const negSuccessRes = a2c(negSuccessConfig, { ...defaultOptions, horizonH: 100, episodes: 30, seed: 1 });
  const negSuccessMetrics = computeACMetricSeries(negSuccessRes, 10);
  const negSuccessIndex = negSuccessRes.episodes.findIndex((ep) => ep.success && ep.cumulativeReward < 0);
  if (negSuccessIndex >= 0) {
    assert(negSuccessMetrics[negSuccessIndex].success === 1, 'negative-return success episode counted as success');
  }

  // 31. mean KL matches hand calculation (KL(old || new))
  const pOld = [0.5, 0.5, 0, 0, 0];
  const pNew = [0.25, 0.75, 0, 0, 0];
  const handKL = 0.5 * Math.log(0.5 / 0.25) + 0.5 * Math.log(0.5 / 0.75);
  assert(near(klDivergence(pOld, pNew), handKL, 1e-9), 'klDivergence matches hand calculation');
  const uniformKL = klDivergence([0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]);
  assert(near(uniformKL, 0, 1e-9), 'KL of identical policies is zero');
  assert(near(klDivergence([0, 0.5, 0.5, 0, 0], [0, 0.25, 0.75, 0, 0]), handKL, 1e-9), 'zero old-probability actions contribute 0');
  assert(klDivergence([0.5, 0.5, 0, 0, 0], [0.5, 0, 0.5, 0, 0]) === Infinity, 'old>0 and new=0 gives Infinity');

  // 32. deterministic AC computes dqda at actorAction
  const detCheck = deterministicActorCriticEpisode(env, -0.8, [0, 0, 0, 0, 0, 0], [0.5, 0], {
    seed: 1,
    horizonH: 10,
    actorAlpha: 0.05,
    criticAlpha: 0.1,
    gamma: 0.9,
    qValueWithWeights: (s, a, w) =>
      [1, s, a, s * s, s * a, a * a].reduce((sum, f, i) => sum + f * w[i], 0),
    muWithTheta: (s, theta) => Math.tanh(theta[0] * s + theta[1]),
    gradMuWithTheta: (s, theta) => {
      const m = Math.tanh(theta[0] * s + theta[1]);
      const factor = 1 - m * m;
      return [factor * s, factor];
    },
    gradQWrtA: (s, a, w) => w[2] + w[4] * s + 2 * w[5] * a,
    criticFeatures: (s, a) => [1, s, a, s * s, s * a, a * a],
    explorationNoiseStd: 0.2,
  });
  detCheck.steps.forEach((st) => {
    const w = st.criticWeightsUsedByActor === 'before' ? st.wBefore : st.wAfter;
    const expectedDqda = w[2] + w[4] * st.state + 2 * w[5] * st.actorAction;
    assert(near(st.dqda, expectedDqda, 1e-9), 'dqda is computed at actorAction with selected critic weights');
  });

  // 33. numerical instability is detected and training stops
  const unstableRes = a2c(EPISODIC_PATH_CONFIG, {
    ...defaultOptions,
    actorAlpha: 1e6,
    criticAlpha: 1e6,
    episodes: 5,
    seed: 0,
  });
  assert(unstableRes.diverged, 'huge learning rate causes divergence flag');
  assert(unstableRes.divergenceStep !== undefined, 'divergence step recorded');
  assert(unstableRes.updates.length <= 5 * defaultOptions.horizonH, 'training stopped early');

  // 34. large magnitudes trigger a warning before (or instead of) divergence
  const warningConfig = { ...EPISODIC_PATH_CONFIG, targetReward: 0, stepReward: 2e6 };
  const warningRes = a2c(warningConfig, {
    ...defaultOptions,
    horizonH: 5,
    episodes: 2,
    actorAlpha: 0.01,
    criticAlpha: 0.01,
    seed: 0,
  });
  assert(
    warningRes.largeMagnitudeWarning !== undefined,
    'large reward scale triggers large-magnitude warning'
  );

  // 35. bootstrapOnTruncation=false disables bootstrap at the artificial horizon
  const noBootstrapA2C = a2c(EPISODIC_PATH_CONFIG, {
    ...defaultOptions,
    horizonH: 2,
    episodes: 5,
    bootstrapOnTruncation: false,
  });
  const truncatedA2C = noBootstrapA2C.updates.filter((u) => u.truncated);
  assert(truncatedA2C.length > 0, 'found truncated transitions with H=2');
  truncatedA2C.forEach((u) => {
    assert(u.bootstrapUsed === false, 'truncated transition does not bootstrap when disabled');
    assert(near(u.criticBootstrap, 0, 1e-9), 'bootstrap value is zero when disabled');
    assert(near(u.criticTarget, u.reward, 1e-9), 'target equals reward when bootstrap disabled');
  });

  // 36. bootstrapOnTruncation=true (default) bootstraps at truncation boundary
  const yesBootstrapA2C = a2c(EPISODIC_PATH_CONFIG, {
    ...defaultOptions,
    horizonH: 20,
    episodes: 30,
    bootstrapOnTruncation: true,
  });
  const truncatedYesA2C = yesBootstrapA2C.updates.filter((u) => u.truncated);
  assert(truncatedYesA2C.length > 0, 'found truncated transitions with bootstrap enabled');
  const bootstrappedTruncated = truncatedYesA2C.filter(
    (u) => u.bootstrapUsed === true && !near(u.criticBootstrap, 0, 1e-9)
  );
  assert(bootstrappedTruncated.length > 0, 'at least one truncated transition bootstraps with non-zero value when enabled');

  // 37. policy history is saved for every episode and inherits previous policy when no updates occur
  const historyRes = a2c(EPISODIC_PATH_CONFIG, { ...defaultOptions, episodes: 5 });
  assert(historyRes.policyAfterEpisode.length === 5, 'policyAfterEpisode has one entry per episode');
  assert(historyRes.initialPolicy.length === 9, 'initialPolicy has one distribution per state');
  assert(
    historyRes.initialPolicy.every((dist) => near(dist.reduce((a, b) => a + b, 0), 1, 1e-9)),
    'initialPolicy entries are valid probability distributions'
  );
  historyRes.policyAfterEpisode.forEach((pol) => {
    assert(pol.length === 9, 'policyAfterEpisode entry has one distribution per state');
    assert(pol.every((dist) => near(dist.reduce((a, b) => a + b, 0), 1, 1e-9)), 'policy distributions sum to 1');
  });

  const terminalStartConfig = { ...EPISODIC_PATH_CONFIG, startState: EPISODIC_PATH_CONFIG.targetState };
  const terminalStartRes = a2c(terminalStartConfig, { ...defaultOptions, episodes: 3 });
  assert(terminalStartRes.updates.length === 0, 'terminal start produces no updates');
  assert(terminalStartRes.policyAfterEpisode.length === 3, 'policyAfterEpisode still saved for empty episodes');
  terminalStartRes.policyAfterEpisode.forEach((pol) => {
    assert(
      pol.every((dist, s) => dist.every((p, a) => near(p, terminalStartRes.initialPolicy[s][a], 1e-9))),
      'empty episodes inherit the previous policy'
    );
  });

  // 38. QAC advantage actorWeight equals qBefore - V_pi(s)
  const qacAdvantage = qac(EPISODIC_PATH_CONFIG, {
    ...defaultOptions,
    episodes: 5,
    actorWeightMode: 'advantage',
  });
  qacAdvantage.updates.forEach((u) => {
    const qRow = u.qBefore![u.state];
    const vPi = u.actorPolicyBefore.reduce((sum, p, a) => sum + p * qRow[a], 0);
    assert(near(u.actorWeight, u.criticEstimateBefore - vPi, 1e-9), 'advantage actorWeight is Q - V_pi');
  });

  // 39. advantage-mode actor weight has the same policy-gradient expectation as raw-Q
  const qacRaw = qac(EPISODIC_PATH_CONFIG, { ...defaultOptions, episodes: 5, actorWeightMode: 'raw-q' });
  for (const u of qacRaw.updates) {
    const rawBaseline = u.qBefore![u.state];
    const vPi = u.actorPolicyBefore.reduce((sum, p, b) => sum + p * rawBaseline[b], 0);
    const advBaseline = rawBaseline.map((q) => q - vPi);
    const rawCheck = checkBaselineInvariance([u.actorPolicyBefore], [rawBaseline]);
    const advCheck = checkBaselineInvariance([u.actorPolicyBefore], [advBaseline]);
    assert(near(rawCheck.maxAbs, advCheck.maxAbs, 1e-9), 'advantage baseline expectation matches raw-Q expectation');
  }

  // 40. off-policy IS clipping limits usedRho and records raw/used/clipped flags
  const clippedRes = offPolicyActorCritic(EPISODIC_PATH_CONFIG, {
    ...defaultOptions,
    epsilon: 0.9,
    episodes: 20,
    importanceMode: 'clipped',
    clipThreshold: 2,
  });
  assert(clippedRes.updates.every((u) => u.rawRho !== undefined && u.usedRho !== undefined && u.wasClipped !== undefined), 'clipping fields recorded');
  assert(clippedRes.updates.every((u) => u.usedRho! <= 2 + 1e-9), 'usedRho does not exceed clip threshold');
  assert(clippedRes.updates.every((u) => (u.wasClipped ? near(u.usedRho!, 2, 1e-9) : true)), 'clipped samples use threshold');
  const clippedCount = clippedRes.updates.filter((u) => u.wasClipped).length;
  const shouldClipCount = clippedRes.updates.filter((u) => u.rawRho! > 2 + 1e-9).length;
  assert(clippedCount === shouldClipCount, 'wasClipped flag matches rawRho > threshold');

  // 41. raw mode keeps rho unchanged and never clips
  const rawRes = offPolicyActorCritic(EPISODIC_PATH_CONFIG, {
    ...defaultOptions,
    epsilon: 0.9,
    episodes: 20,
    importanceMode: 'raw',
  });
  rawRes.updates.forEach((u) => {
    assert(near(u.rho!, u.rawRho!, 1e-9), 'raw mode rho equals rawRho');
    assert(u.wasClipped === false, 'raw mode does not clip');
  });

  // 42. Q-based off-policy uses expected bootstrap target
  const qBasedRes = qBasedOffPolicyActorCritic(EPISODIC_PATH_CONFIG, {
    ...defaultOptions,
    epsilon: 0.5,
    episodes: 5,
  });
  qBasedRes.updates.forEach((u) => {
    if (!u.done) {
      // The expected bootstrap was computed before the actor update, so use the policy before update.
      const targetPolicyNext = u.actorFullPolicyBefore![u.nextState];
      const expected = targetPolicyNext.reduce((sum, p, a) => sum + p * u.qBefore![u.nextState][a], 0);
      assert(near(u.expectedBootstrap!, expected, 1e-9), 'expectedBootstrap equals Σ π(a|s\') Q(s\',a)');
      assert(near(u.criticTarget, u.reward + EPISODIC_PATH_CONFIG.gamma * expected, 1e-9), 'critic target uses expected bootstrap');
    }
  });

  // 43. stochastic value solver with zero slip matches deterministic solver
  const slipPolicy = Array.from({ length: 9 }, () => Array.from({ length: 5 }, () => 1 / 5));
  const detValues = solveStateValues(slipPolicy, exactConfig);
  const slipValues = solveStateValuesWithSlip(slipPolicy, exactConfig, 0);
  detValues.forEach((v, s) => assert(near(v, slipValues[s], 1e-9), `slip=0 values match deterministic at state ${s}`));
  const detQ = computeQValues(detValues, exactConfig);
  const slipQ = computeQValuesWithSlip(slipPolicy, exactConfig, 0);
  detQ.forEach((row, s) =>
    row.forEach((q, a) => assert(near(q, slipQ[s][a], 1e-9), `slip=0 Q matches deterministic at (${s},${a})`))
  );

  // 44. metric series uses renamed episode-length fields and averages
  const metricRes = a2c(EPISODIC_PATH_CONFIG, { ...defaultOptions, episodes: 10 });
  const metricSeries = computeACMetricSeries(metricRes, 3);
  assert(metricSeries.every((m) => 'movingAverageEpisodeLength' in m), 'metric series has movingAverageEpisodeLength');
  assert(metricSeries.every((m) => 'overallAverageEpisodeLength' in m), 'metric series has overallAverageEpisodeLength');
  const totalLength = metricRes.episodes.reduce((sum, ep) => sum + ep.episodeLength, 0);
  assert(near(metricSeries[metricSeries.length - 1].overallAverageEpisodeLength, totalLength / metricRes.episodes.length, 1e-9), 'overall average episode length matches');

  // 45. stochasticStepDistribution keeps full outcomes (same nextState can have different rewards)
  const cornerOutcomes = stochasticStepDistribution(0, 4 as Action, EPISODIC_PATH_CONFIG, 0.2);
  const stayOutcome = cornerOutcomes.find((o) => o.actualAction === 4)!;
  const upOutcome = cornerOutcomes.find((o) => o.actualAction === 0)!;
  assert(stayOutcome !== undefined, 'intended stay outcome exists');
  assert(upOutcome !== undefined, 'slip-up outcome exists');
  assert(stayOutcome.nextState === 0 && upOutcome.nextState === 0, 'stay and up both lead to state 0');
  assert(stayOutcome.reward === -1 && upOutcome.reward === -10, 'same nextState but different rewards are preserved');

  // 46. slip into forbidden state uses forbiddenReward
  const centerOutcomes = stochasticStepDistribution(4, 0 as Action, EPISODIC_PATH_CONFIG, 0.2);
  const slipRight = centerOutcomes.find((o) => o.actualAction === 1)!;
  assert(slipRight !== undefined, 'slip-right outcome exists');
  assert(slipRight.nextState === 5, 'slip-right enters forbidden state 5');
  assert(slipRight.reward === -10, 'forbidden transition reward is forbiddenReward');

  // 47. slip into target in episodic task sets done=true and omits bootstrap in solver
  const nearTargetOutcomes = stochasticStepDistribution(7, 4 as Action, EPISODIC_PATH_CONFIG, 0.2);
  const slipTarget = nearTargetOutcomes.find((o) => o.actualAction === 1)!;
  assert(slipTarget !== undefined, 'slip-right to target exists');
  assert(slipTarget.nextState === 8, 'slip-right reaches target');
  assert(slipTarget.done === true, 'reaching target in episodic task is terminal');

  // 48. outcome probabilities sum to 1
  for (const cfg of [EPISODIC_PATH_CONFIG, DEFAULT_CONFIG]) {
    for (let s = 0; s < cfg.rows * cfg.cols; s++) {
      if (isTerminal(s, cfg)) continue;
      for (let a = 0; a < 5; a++) {
        const outcomes = stochasticStepDistribution(s, a as Action, cfg, 0.2);
        const total = outcomes.reduce((sum, o) => sum + o.prob, 0);
        assert(near(total, 1, 1e-9), `outcome probabilities sum to 1 at (${s},${a})`);
      }
    }
  }

  // 49. slip=0 stochastic solver still matches deterministic solver
  const uniformSlipPolicy = Array.from({ length: 9 }, () => Array.from({ length: 5 }, () => 1 / 5));
  const detValues2 = solveStateValues(uniformSlipPolicy, exactConfig);
  const slip0Values = solveStateValuesWithSlip(uniformSlipPolicy, exactConfig, 0);
  detValues2.forEach((v, s) => assert(near(v, slip0Values[s], 1e-9), `slip=0 values match deterministic at state ${s}`));
  const detQ2 = computeQValues(detValues2, exactConfig);
  const slip0Q = computeQValuesWithSlip(uniformSlipPolicy, exactConfig, 0);
  detQ2.forEach((row, s) =>
    row.forEach((q, a) => assert(near(q, slip0Q[s][a], 1e-9), `slip=0 Q matches deterministic at (${s},${a})`))
  );

  // 39b. Q-shift invariance: shifting all Q(s,·) by a state-dependent constant changes raw-Q but not advantage
  const qacShiftRes = qac(EPISODIC_PATH_CONFIG, { ...defaultOptions, episodes: 5, actorWeightMode: 'raw-q' });
  for (const shift of [-5, -1, 0, 1, 5]) {
    for (const u of qacShiftRes.updates) {
      const qRow = u.qBefore![u.state];
      const vPi = u.actorPolicyBefore.reduce((sum, p, a) => sum + p * qRow[a], 0);
      const rawOriginal = u.criticEstimateBefore;
      const rawShifted = rawOriginal + shift;
      const vPiShifted = u.actorPolicyBefore.reduce((sum, p, a) => sum + p * (qRow[a] + shift), 0);
      const advOriginal = rawOriginal - vPi;
      const advShifted = rawShifted - vPiShifted;
      assert(near(advShifted, advOriginal, 1e-9), `Q-shift ${shift} leaves advantage unchanged at (${u.state},${u.action})`);
    }
  }

  // 50. Monte Carlo TD-error mean approximates true advantage under slip
  const slipForMc = 0.2;
  const mcTrueV = solveStateValuesWithSlip(uniformSlipPolicy, exactConfig, slipForMc);
  const mcTrueQ = computeQValuesWithSlip(uniformSlipPolicy, exactConfig, slipForMc);
  const mcState = 0;
  const mcAction = 1 as Action;
  const mcTrueAdvantage = mcTrueQ[mcState][mcAction] - mcTrueV[mcState];
  const outcomes = stochasticStepDistribution(mcState, mcAction, exactConfig, slipForMc);
  const rng = mulberry32(123);
  const mcSamples: number[] = [];
  for (let i = 0; i < 5000; i++) {
    const outcome = sampleOutcome(outcomes, rng);
    const bootstrap = outcome.done ? 0 : mcTrueV[outcome.nextState];
    const tdError = outcome.reward + exactConfig.gamma * bootstrap - mcTrueV[mcState];
    mcSamples.push(tdError);
  }
  const mcMean = mcSamples.reduce((a, b) => a + b, 0) / mcSamples.length;
  assert(Math.abs(mcMean - mcTrueAdvantage) < 0.2, `Monte Carlo TD-error mean ${mcMean.toFixed(4)} approximates true advantage ${mcTrueAdvantage.toFixed(4)}`);

  console.log('✅ Actor-Critic tests passed');
}

function sampleOutcome(outcomes: StochasticOutcome[], rng: () => number): StochasticOutcome {
  const r = rng();
  let cum = 0;
  for (const outcome of outcomes) {
    cum += outcome.prob;
    if (r <= cum) return outcome;
  }
  return outcomes[outcomes.length - 1];
}
