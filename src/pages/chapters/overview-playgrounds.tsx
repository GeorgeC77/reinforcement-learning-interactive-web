/**
 * Lightweight interactive previews embedded in chapter overview pages.
 * Each component illustrates the chapter's single most important idea in a
 * compact, self-contained widget that links to the full demos.
 */
import { useMemo, useState, type ReactNode } from 'react';
import { Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import KaTeX from '@/components/KaTeX';
import GridWorld from '@/components/rl/GridWorld';
import LineChart from '@/components/LineChart';
import { mulberry32 } from '@/lib/rl/stochasticApproximation';
import {
  DEFAULT_CONFIG,
  ACTION_NAMES,
  type Action,
  deterministicPolicy,
  iterateStateValues,
  solveStateValues,
  valueIterationConvergence,
  policyIteration,
  transitionTable,
  reward,
  step,
  isTerminal,
} from '@/lib/rl/gridworld';

function Shell({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
      <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
        <Sparkles className="w-5 h-5 text-blue-600" />
        {title}
      </h2>
      {children}
    </section>
  );
}

const PREVIEW_POLICY = deterministicPolicy([1, 1, 2, 1, 1, 2, 4, 1, 4] as Action[]);

// ---------------- Ch1: click a state to inspect its transition model ----------------
export function Ch1Playground() {
  const [selected, setSelected] = useState(0);
  const transitions = transitionTable(selected, DEFAULT_CONFIG);
  return (
    <Shell title="互动预览：点击状态，查看它的转移与奖励模型">
      <div className="grid md:grid-cols-[auto_1fr] gap-6 items-start">
        <GridWorld config={DEFAULT_CONFIG} highlightState={selected} onCellClick={setSelected} className="max-w-[240px]" />
        <div className="text-sm text-gray-700">
          <div className="font-semibold mb-2">
            状态 s{selected + 1}
            {isTerminal(selected, DEFAULT_CONFIG) && <span className="ml-2 text-xs text-green-600">（终止状态）</span>}
          </div>
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="text-gray-500 border-b"><th className="text-left py-1">动作</th><th className="text-left">下一状态</th><th className="text-left">奖励</th></tr>
            </thead>
            <tbody>
              {transitions.map((nextState, a) => (
                <tr key={a} className="border-b last:border-0">
                  <td className="py-1">{ACTION_NAMES[a]}</td>
                  <td>s{nextState + 1}</td>
                  <td className="font-mono">{reward(selected, DEFAULT_CONFIG).toFixed(0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-gray-500">
            马尔可夫性：下一状态与奖励只由当前 (s,a) 决定，与历史无关。
          </p>
        </div>
      </div>
    </Shell>
  );
}

// ---------------- Ch2: step through policy-evaluation iterations ----------------
export function Ch2Playground() {
  const [k, setK] = useState(0);
  const history = useMemo(() => iterateStateValues(PREVIEW_POLICY, DEFAULT_CONFIG, 40), []);
  const maxK = history.length - 1;
  const safeK = Math.min(k, maxK);
  return (
    <Shell title="互动预览：贝尔曼迭代 v ← rπ + γPπv 逐步收敛">
      <div className="grid md:grid-cols-[auto_1fr] gap-6 items-center">
        <GridWorld config={DEFAULT_CONFIG} values={history[safeK]} showValues className="max-w-[240px]" />
        <div>
          <div className="flex items-center gap-3 mb-3">
            <Button size="sm" variant="outline" disabled={safeK === 0} onClick={() => setK(safeK - 1)}>上一步</Button>
            <Button size="sm" disabled={safeK === maxK} onClick={() => setK(safeK + 1)}>备份一步</Button>
            <Button size="sm" variant="ghost" onClick={() => setK(0)}>重置</Button>
          </div>
          <input type="range" min={0} max={maxK} value={safeK} onChange={(e) => setK(Number(e.target.value))} className="w-full mb-2" />
          <div className="text-sm text-gray-700">
            第 {safeK} 次迭代 · v(s1) = <span className="font-mono font-semibold">{history[safeK][0].toFixed(3)}</span>
            <span className="text-gray-500">（真值 {solveStateValues(PREVIEW_POLICY, DEFAULT_CONFIG)[0].toFixed(3)}）</span>
          </div>
          <p className="mt-2 text-xs text-gray-500">值函数从全零逐步逼近贝尔曼方程的唯一解。</p>
        </div>
      </div>
    </Shell>
  );
}

// ---------------- Ch3: contraction — error decays geometrically ----------------
export function Ch3Playground() {
  const [gamma, setGamma] = useState(DEFAULT_CONFIG.gamma);
  const data = useMemo(() => {
    const conv = valueIterationConvergence({ ...DEFAULT_CONFIG, gamma }, 100, 1e-8);
    const e0 = conv.errors[0] ?? 1;
    return conv.errors.slice(0, 60).map((err, k) => ({
      k,
      error: Math.max(err, 1e-10),
      bound: Math.max(e0 * Math.pow(gamma, k), 1e-10),
    }));
  }, [gamma]);
  return (
    <Shell title="互动预览：γ 越小，贝尔曼最优算子压缩得越快">
      <div className="flex items-center gap-3 mb-2">
        <span className="text-sm text-gray-700">γ</span>
        <div className="flex-1"><Slider value={[gamma]} min={0.1} max={0.99} step={0.01} onValueChange={([v]) => setGamma(v)} /></div>
        <span className="font-mono text-sm w-10 text-right">{gamma.toFixed(2)}</span>
      </div>
      <LineChart data={data} xKey="k" xLabel="迭代 k" yLabel="‖v_k − v*‖∞（log）" logY height={200}
        series={[{ key: 'error', name: '实际误差', color: '#2563eb' }, { key: 'bound', name: '上界 γᵏ‖v₀−v*‖', color: '#f59e0b', strokeDasharray: '6 3' }]} />
    </Shell>
  );
}

// ---------------- Ch4: VI vs PI convergence speed ----------------
export function Ch4Playground() {
  const data = useMemo(() => {
    const vi = valueIterationConvergence(DEFAULT_CONFIG, 200, 1e-6);
    const pi = policyIteration(DEFAULT_CONFIG, 100, 100, 1e-6);
    const viErr = vi.errors;
    const piFirst = pi.values[0];
    const piStar = pi.values[pi.values.length - 1];
    const maxLen = Math.max(viErr.length, 12);
    const rows: Record<string, number>[] = [];
    const errOf = (v: number[]) => Math.max(...v.map((x, i) => Math.abs(x - piStar[i])));
    for (let k = 0; k < maxLen; k++) {
      rows.push({
        k,
        vi: k < viErr.length ? Math.max(viErr[k], 1e-10) : undefined as unknown as number,
        pi: k < pi.values.length ? Math.max(errOf(pi.values[k]), 1e-10) : (piFirst ? Math.max(errOf(piStar), 1e-10) : 0),
      });
    }
    return rows;
  }, []);
  return (
    <Shell title="互动预览：策略迭代通常比值迭代用更少轮数收敛">
      <LineChart data={data} xKey="k" xLabel="外层迭代轮数" yLabel="‖v_k − v*‖∞（log）" logY height={200}
        series={[{ key: 'vi', name: '值迭代 VI', color: '#2563eb' }, { key: 'pi', name: '策略迭代 PI', color: '#ef4444' }]} />
      <p className="mt-2 text-xs text-gray-500">PI 每轮做精确策略评估，单轮成本高但轮数少；VI 每轮便宜但轮数多。</p>
    </Shell>
  );
}

// ---------------- Ch5: sample average converges to the expectation ----------------
export function Ch5Playground() {
  const [n, setN] = useState(0);
  const [seed, setSeed] = useState(1);
  const TRUE_MEAN = 0.7;
  const data = useMemo(() => {
    const rng = mulberry32(seed);
    const rows: { n: number; mean: number; truth: number }[] = [];
    let sum = 0;
    for (let i = 1; i <= 200; i++) {
      // Sample a return: target reward with noise.
      sum += TRUE_MEAN + (rng() - 0.5) * 2;
      rows.push({ n: i, mean: sum / i, truth: TRUE_MEAN });
    }
    return rows;
  }, [seed]);
  const safeN = Math.max(1, Math.min(n, 200));
  return (
    <Shell title="互动预览：样本均值随回合数增加收敛到期望回报">
      <div className="flex items-center gap-3 mb-2">
        <Button size="sm" onClick={() => setN((v) => Math.min(200, v + 1))}>采样 1 回合</Button>
        <Button size="sm" variant="outline" onClick={() => setN((v) => Math.min(200, v + 10))}>+10</Button>
        <Button size="sm" variant="ghost" onClick={() => { setN(0); setSeed((s) => s + 1); }}>换一批</Button>
        <span className="text-sm text-gray-700">已采样 {safeN} 回合 · 均值 {data[safeN - 1].mean.toFixed(3)}</span>
      </div>
      <LineChart data={data.slice(0, safeN)} xKey="n" xLabel="回合数" yLabel="样本均值" height={200}
        series={[{ key: 'mean', name: '样本均值', color: '#2563eb' }, { key: 'truth', name: '期望回报', color: '#ef4444', strokeDasharray: '6 3' }]} />
    </Shell>
  );
}

// ---------------- Ch6: step-size conditions Σα=∞, Σα²<∞ ----------------
export function Ch6Playground() {
  const [p, setP] = useState(0.8);
  const sums = useMemo(() => {
    let s1 = 0, s2 = 0;
    for (let k = 1; k <= 5000; k++) {
      const a = 1 / Math.pow(k, p);
      s1 += a; s2 += a * a;
    }
    return { s1, s2 };
  }, [p]);
  const ok = sums.s1 > 1e6 && sums.s2 < 1e6;
  return (
    <Shell title="互动预览：Robbins–Monro 步长条件 α_k = 1/k^p">
      <div className="flex items-center gap-3 mb-3">
        <span className="text-sm text-gray-700">p</span>
        <div className="flex-1"><Slider value={[p]} min={0.1} max={1.5} step={0.05} onValueChange={([v]) => setP(v)} /></div>
        <span className="font-mono text-sm w-10 text-right">{p.toFixed(2)}</span>
      </div>
      <div className="grid grid-cols-3 gap-3 text-center text-sm">
        <div className="bg-gray-50 rounded p-3 border border-gray-200">
          <div className="text-xs text-gray-500 mb-1">Σα_k</div>
          <div className={`font-mono font-semibold ${sums.s1 > 1e6 ? 'text-green-600' : 'text-red-600'}`}>{sums.s1 > 1e6 ? '→ ∞ ✓' : sums.s1.toExponential(2)}</div>
        </div>
        <div className="bg-gray-50 rounded p-3 border border-gray-200">
          <div className="text-xs text-gray-500 mb-1">Σα_k²</div>
          <div className={`font-mono font-semibold ${sums.s2 < 1e6 ? 'text-green-600' : 'text-red-600'}`}>{sums.s2 < 1e6 ? '收敛 ✓' : '发散 ✗'}</div>
        </div>
        <div className={`rounded p-3 border ${ok ? 'bg-green-50 border-green-200' : 'bg-amber-50 border-amber-200'}`}>
          <div className="text-xs text-gray-500 mb-1">收敛条件</div>
          <div className={`font-semibold ${ok ? 'text-green-700' : 'text-amber-700'}`}>{ok ? '满足' : '不满足'}</div>
        </div>
      </div>
      <p className="mt-2 text-xs text-gray-500">需要 Σα_k = ∞（足够学习）且 Σα_k² &lt; ∞（噪声衰减）。p=1 是经典选择。</p>
    </Shell>
  );
}

// ---------------- Ch7: one TD(0) update ----------------
export function Ch7Playground() {
  const [alpha, setAlpha] = useState(0.5);
  const [v, setV] = useState<number[]>(() => new Array(9).fill(0));
  const [s, setS] = useState(0);
  const [a, setA] = useState<Action>(1);
  const result = step(s, a, DEFAULT_CONFIG);
  const delta = result.reward + DEFAULT_CONFIG.gamma * v[result.nextState] - v[s];
  function apply() {
    const nv = [...v];
    nv[s] = v[s] + alpha * delta;
    setV(nv);
  }
  function reset() { setV(new Array(9).fill(0)); }
  return (
    <Shell title="互动预览：一条转移如何更新值函数">
      <div className="grid md:grid-cols-[auto_1fr] gap-6 items-center">
        <GridWorld config={DEFAULT_CONFIG} values={v} showValues highlightState={s} highlightNextState={result.nextState} className="max-w-[240px]" />
        <div className="text-sm text-gray-700 space-y-2">
          <div className="flex gap-2 flex-wrap">
            <select value={s} onChange={(e) => setS(Number(e.target.value))} className="border rounded px-2 py-1">
              {Array.from({ length: 8 }, (_, i) => <option key={i} value={i}>s{i + 1}</option>)}
            </select>
            <select value={a} onChange={(e) => setA(Number(e.target.value) as Action)} className="border rounded px-2 py-1">
              {ACTION_NAMES.map((n, i) => <option key={i} value={i}>{n}</option>)}
            </select>
            <span className="text-gray-500">→ s{result.nextState + 1}，r = {result.reward}</span>
          </div>
          <div>TD 误差 <KaTeX math={String.raw`\delta`} /> = r + γv(s′) − v(s) = <span className="font-mono font-semibold">{delta.toFixed(3)}</span></div>
          <div className="flex items-center gap-3">
            <span className="text-gray-600">α</span>
            <div className="flex-1"><Slider value={[alpha]} min={0.05} max={1} step={0.05} onValueChange={([x]) => setAlpha(x)} /></div>
            <span className="font-mono w-10 text-right">{alpha.toFixed(2)}</span>
          </div>
          <div className="flex gap-2">
            <Button size="sm" onClick={apply}>更新 v(s)</Button>
            <Button size="sm" variant="outline" onClick={reset}>清零</Button>
          </div>
          <p className="text-xs text-gray-500">v(s) ← v(s) + α·δ。与 MC 不同，TD 用 bootstrap 目标 r + γv(s′) 立即更新。</p>
        </div>
      </div>
    </Shell>
  );
}

// ---------------- Ch8: polynomial curve fitting ----------------
export function Ch8Playground() {
  const [degree, setDegree] = useState(2);
  const fit = useMemo(() => {
    // Fit y = sin(2x) on [0, π] with a polynomial of the given degree (least squares).
    const xs = Array.from({ length: 25 }, (_, i) => (i / 24) * Math.PI);
    const ys = xs.map((x) => Math.sin(2 * x));
    const d = degree + 1;
    // Normal equations for polynomial least squares.
    const A: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
    const b = new Array(d).fill(0);
    for (let i = 0; i < xs.length; i++) {
      const powers = Array.from({ length: d }, (_, j) => Math.pow(xs[i], j));
      for (let r = 0; r < d; r++) {
        for (let c = 0; c < d; c++) A[r][c] += powers[r] * powers[c];
        b[r] += powers[r] * ys[i];
      }
    }
    // Gaussian elimination.
    for (let col = 0; col < d; col++) {
      let piv = col;
      for (let r = col + 1; r < d; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
      [A[col], A[piv]] = [A[piv], A[col]];
      [b[col], b[piv]] = [b[piv], b[col]];
      const diag = A[col][col] || 1e-12;
      for (let r = 0; r < d; r++) {
        if (r === col) continue;
        const f = A[r][col] / diag;
        for (let c = 0; c < d; c++) A[r][c] -= f * A[col][c];
        b[r] -= f * b[col];
      }
    }
    const w = b.map((v, i) => v / (A[i][i] || 1e-12));
    return xs.map((x, i) => ({
      x: Number(x.toFixed(3)),
      true: ys[i],
      fit: w.reduce((s, c, j) => s + c * Math.pow(x, j), 0),
    }));
  }, [degree]);
  return (
    <Shell title="互动预览：用少量参数逼近复杂函数——函数近似的核心思想">
      <div className="flex items-center gap-3 mb-2">
        <span className="text-sm text-gray-700">多项式阶数</span>
        <div className="flex-1"><Slider value={[degree]} min={1} max={8} step={1} onValueChange={([v]) => setDegree(v)} /></div>
        <span className="font-mono text-sm w-10 text-right">{degree}</span>
      </div>
      <LineChart data={fit} xKey="x" xLabel="x" yLabel="y" height={200}
        series={[{ key: 'true', name: '目标 y = sin(2x)', color: '#2563eb' }, { key: 'fit', name: `${degree} 阶多项式拟合`, color: '#ef4444', strokeDasharray: '5 3' }]} />
      <p className="mt-2 text-xs text-gray-500">阶数越高拟合越准，但参数也越多——这正是值函数近似中的表达能力/泛化权衡。</p>
    </Shell>
  );
}

// ---------------- Ch9: softmax with temperature ----------------
export function Ch9Playground() {
  const [tau, setTau] = useState(1);
  const logits = [2.0, 1.0, 0.1];
  const probs = useMemo(() => {
    const scaled = logits.map((l) => l / tau);
    const m = Math.max(...scaled);
    const exps = scaled.map((l) => Math.exp(l - m));
    const sum = exps.reduce((s, x) => s + x, 0);
    return exps.map((x) => x / sum);
  }, [tau]);
  return (
    <Shell title="互动预览：softmax 把 logits 变成动作概率">
      <div className="flex items-center gap-3 mb-3">
        <span className="text-sm text-gray-700">温度 τ</span>
        <div className="flex-1"><Slider value={[tau]} min={0.2} max={5} step={0.1} onValueChange={([v]) => setTau(v)} /></div>
        <span className="font-mono text-sm w-10 text-right">{tau.toFixed(1)}</span>
      </div>
      <div className="space-y-2">
        {probs.map((p, i) => (
          <div key={i} className="flex items-center gap-2 text-sm">
            <span className="w-16 text-gray-600">a{i + 1}</span>
            <div className="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
              <div className="h-full bg-blue-500 transition-all" style={{ width: `${p * 100}%` }} />
            </div>
            <span className="w-14 text-right font-mono">{p.toFixed(3)}</span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-xs text-gray-500">τ → 0 趋向 argmax（贪心）；τ → ∞ 趋向均匀（随机探索）。策略梯度直接对 logits 的参数 θ 求梯度。</p>
    </Shell>
  );
}

// ---------------- Ch10: actor-critic one-step update ----------------
export function Ch10Playground() {
  const [alphaTheta, setAlphaTheta] = useState(0.1);
  const [alphaW, setAlphaW] = useState(0.5);
  const [theta, setTheta] = useState(0);
  const [w, setW] = useState(0);
  const TARGET = 1;
  const delta = TARGET - w; // critic's TD error (simplified)
  function stepOnce() {
    setTheta((t) => t + alphaTheta * delta);
    setW((v) => v + alphaW * delta);
  }
  function reset() { setTheta(0); setW(0); }
  return (
    <Shell title="互动预览：Critic 评估、Actor 改进——同一个 TD 误差驱动两者">
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-gray-50 rounded p-3 border border-gray-200">
          <div className="text-xs text-gray-500 mb-1">Critic（价值 w）</div>
          <div className="font-mono text-lg font-semibold">{w.toFixed(3)}</div>
          <div className="text-xs text-gray-500">目标 v = {TARGET}</div>
          <div className="flex items-center gap-2 mt-2">
            <span className="text-xs text-gray-600">α_w</span>
            <div className="flex-1"><Slider value={[alphaW]} min={0.05} max={1} step={0.05} onValueChange={([v]) => setAlphaW(v)} /></div>
            <span className="font-mono text-xs w-8">{alphaW.toFixed(2)}</span>
          </div>
        </div>
        <div className="bg-gray-50 rounded p-3 border border-gray-200">
          <div className="text-xs text-gray-500 mb-1">Actor（偏好 θ）</div>
          <div className="font-mono text-lg font-semibold">{theta.toFixed(3)}</div>
          <div className="text-xs text-gray-500">TD 误差 δ = {delta.toFixed(3)}</div>
          <div className="flex items-center gap-2 mt-2">
            <span className="text-xs text-gray-600">α_θ</span>
            <div className="flex-1"><Slider value={[alphaTheta]} min={0.05} max={1} step={0.05} onValueChange={([v]) => setAlphaTheta(v)} /></div>
            <span className="font-mono text-xs w-8">{alphaTheta.toFixed(2)}</span>
          </div>
        </div>
      </div>
      <div className="flex gap-2 mt-3">
        <Button size="sm" onClick={stepOnce}>联合更新一步</Button>
        <Button size="sm" variant="outline" onClick={reset}>重置</Button>
      </div>
      <p className="mt-2 text-xs text-gray-500">w ← w + α_w·δ（Critic 学习价值）；θ ← θ + α_θ·δ·∇logπ（Actor 沿梯度改进）。</p>
    </Shell>
  );
}
