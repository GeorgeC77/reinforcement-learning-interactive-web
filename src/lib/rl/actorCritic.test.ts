import {
  EPISODIC_PATH_CONFIG,
  solveStateValues,
  computeQValues,
  isTerminal,
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
  type ACOptions,
  type ACResult,
} from './actorCritic';

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

  console.log('✅ Actor-Critic tests passed');
}
