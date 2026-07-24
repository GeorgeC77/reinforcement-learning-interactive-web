import { dynaQ } from './dyna';
import { EPISODIC_PATH_CONFIG } from './gridworld';

function assert(cond: boolean, msg: string): asserts cond {
  if (!cond) throw new Error(`Test failed: ${msg}`);
}

export function runDynaTests() {
  const base = {
    episodes: 60,
    maxSteps: 50,
    alpha: 0.2,
    epsilon: 0.1,
  };

  // 1. Reproducibility.
  const a = dynaQ(EPISODIC_PATH_CONFIG, { ...base, planningSteps: 5, seed: 3 });
  const b = dynaQ(EPISODIC_PATH_CONFIG, { ...base, planningSteps: 5, seed: 3 });
  assert(
    a.episodes.every((e, i) => e.steps === b.episodes[i].steps && e.cumulativeReward === b.episodes[i].cumulativeReward),
    'same seed should produce identical episodes'
  );

  // 2. planningSteps = 0 performs no planning updates.
  const noPlan = dynaQ(EPISODIC_PATH_CONFIG, { ...base, planningSteps: 0, seed: 3 });
  assert(noPlan.totalPlanningSteps === 0, 'planningSteps=0 should skip planning');
  assert(noPlan.totalEnvSteps > 0, 'should still take real env steps');

  // 3. Dyna with planning reaches low q* RMSE in fewer env episodes than no-plan.
  const dyna = dynaQ(EPISODIC_PATH_CONFIG, { ...base, planningSteps: 10, seed: 11 });
  const plain = dynaQ(EPISODIC_PATH_CONFIG, { ...base, planningSteps: 0, seed: 11 });
  const dynaFinalRmse = dyna.episodes[dyna.episodes.length - 1].qRmse;
  const plainFinalRmse = plain.episodes[plain.episodes.length - 1].qRmse;
  assert(
    dynaFinalRmse <= plainFinalRmse + 1e-9,
    `Dyna (planning=10) should converge at least as fast as Q-learning (dyna ${dynaFinalRmse.toFixed(3)} vs plain ${plainFinalRmse.toFixed(3)})`
  );

  // 4. The learned model becomes accurate on visited (s,a) pairs.
  const last = dyna.episodes[dyna.episodes.length - 1];
  assert(last.modelAccuracy > 0.9, `model should be accurate on most visited pairs (got ${last.modelAccuracy.toFixed(2)})`);

  // 5. Episodes get shorter as the policy improves on the path-finding task.
  const early = dyna.episodes.slice(0, 10).reduce((s, e) => s + e.steps, 0) / 10;
  const late = dyna.episodes.slice(-10).reduce((s, e) => s + e.steps, 0) / 10;
  assert(late <= early, `episodes should shorten with learning (early ${early.toFixed(1)} vs late ${late.toFixed(1)})`);

  // 6. Planning updates accumulate.
  assert(dyna.totalPlanningSteps === dyna.totalEnvSteps * 10, 'planning steps should be 10x env steps');

  console.log('All Dyna-Q tests passed.');
}
