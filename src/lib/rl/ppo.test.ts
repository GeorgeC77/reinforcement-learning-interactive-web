import { clipRatioObjective, clipObjectiveCurve, runPpoToy } from './ppo';

function assert(cond: boolean, msg: string): asserts cond {
  if (!cond) throw new Error(`Test failed: ${msg}`);
}

export function runPpoTests() {
  // 1. Inside the clip range both objectives agree.
  const inside = clipRatioObjective(1.1, 2, 0.2);
  assert(Math.abs(inside.unclipped - inside.clipped) < 1e-12, 'inside [1-ε,1+ε] objectives should match');

  // 2. Outside the clip range with A > 0, the clipped objective stops growing.
  const outPos = clipRatioObjective(1.9, 1, 0.2);
  assert(Math.abs(outPos.clipped - 1.2) < 1e-9, 'A>0, r>1+ε: clipped should be (1+ε)·A');
  assert(outPos.unclipped > outPos.clipped, 'unclipped should keep growing with r');

  // 3. Outside the clip range with A < 0 (r < 1-ε), the clipped objective floors at (1-ε)·A.
  const outNeg = clipRatioObjective(0.5, -1, 0.2);
  assert(Math.abs(outNeg.clipped - -0.8) < 1e-9, 'A<0, r<1-ε: clipped should be (1-ε)·A');

  // 4. The curve is clipped (bounded) for extreme ratios on both sides.
  const curve = clipObjectiveCurve(1, 0.2);
  const maxClipped = Math.max(...curve.map((d) => d.clipped));
  assert(maxClipped <= 1.2 + 1e-9, 'clipped objective should be bounded by (1+ε)·A');

  // 5. Clipped training converges toward the better action without overshooting ratio bounds as wildly.
  const clipped = runPpoToy({ alpha: 1.0, epsilon: 0.2, iterations: 60, mode: 'clipped' });
  const unclipped = runPpoToy({ alpha: 1.0, epsilon: 0.2, iterations: 60, mode: 'unclipped' });
  const finalClipped = clipped[clipped.length - 1];
  const finalUnclipped = unclipped[unclipped.length - 1];
  assert(finalClipped.p1 > 0.8, `clipped should learn to prefer the better action (p1=${finalClipped.p1.toFixed(2)})`);
  assert(finalUnclipped.p1 > 0.8, `unclipped should also prefer the better action (p1=${finalUnclipped.p1.toFixed(2)})`);
  const maxRatioClipped = Math.max(...clipped.map((s) => s.maxRatio));
  const maxRatioUnclipped = Math.max(...unclipped.map((s) => s.maxRatio));
  assert(
    maxRatioClipped <= maxRatioUnclipped + 1e-9,
    `clipped updates should not exceed unclipped ratio excursions (clipped ${maxRatioClipped.toFixed(2)} vs unclipped ${maxRatioUnclipped.toFixed(2)})`
  );

  // 6. Deterministic: same options give identical histories.
  const again = runPpoToy({ alpha: 1.0, epsilon: 0.2, iterations: 20, mode: 'clipped' });
  assert(
    again.every((s, i) => s.p1 === clipped[i].p1),
    'same options should reproduce identical training curves'
  );

  console.log('All PPO tests passed.');
}
