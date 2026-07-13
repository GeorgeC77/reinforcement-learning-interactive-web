import {
  EPISODIC_PATH_CONFIG,
  solveStateValues,
  computeQValues,
} from './gridworld';
import {
  qac,
  a2c,
  offPolicyActorCritic,
  deterministicActorCriticEpisode,
  checkCoverage,
  effectiveSampleSize,
  policyWeightedStateValues,
  type ACOptions,
} from './actorCritic';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(`Test failed: ${msg}`);
}

function near(a: number, b: number, eps = 1e-9) {
  return Math.abs(a - b) <= eps;
}

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

  // 9. A2C TD error formula
  const a2cRes = a2c(EPISODIC_PATH_CONFIG, defaultOptions);
  a2cRes.updates.forEach((u, i) => {
    const expected = u.reward + EPISODIC_PATH_CONFIG.gamma * u.criticBootstrap - u.criticEstimateBefore;
    assert(near(u.tdError, expected, 1e-9), `a2c tdError formula ${i}`);
  });

  // 10. With exact V, mean TD error approximates true advantage
  const exactConfig = { ...EPISODIC_PATH_CONFIG, taskType: 'continuing' as const };
  const uniformPolicy = Array.from({ length: 9 }, () => Array.from({ length: 5 }, () => 1 / 5));
  const trueV = solveStateValues(uniformPolicy, exactConfig);
  const trueQ = computeQValues(trueV, exactConfig);
  const testState = 0;
  const testAction = 1;
  const deltas: number[] = [];
  for (let seed = 1; seed <= 500; seed++) {
    const res = a2c(exactConfig, {
      seed,
      horizonH: 1,
      actorAlpha: 0,
      criticAlpha: 0,
      episodes: 1,
    });
    const u = res.updates.find((up) => up.state === testState && up.action === testAction);
    if (u) deltas.push(u.reward + exactConfig.gamma * (u.done ? 0 : trueV[u.nextState]) - trueV[testState]);
  }
  assert(deltas.length > 50, 'collected enough exact-critic samples');
  const meanDelta = deltas.reduce((a, b) => a + b, 0) / deltas.length;
  const trueAdvantage = trueQ[testState][testAction] - trueV[testState];
  assert(near(meanDelta, trueAdvantage, 0.2), 'mean TD error close to true advantage with exact critic');

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

  console.log('✅ Actor-Critic tests passed');
}
