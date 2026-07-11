import { useState, useMemo, useEffect } from 'react';
import { TrendingDown, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import LineChart from '@/components/LineChart';
import ConceptAccordion from '@/components/ConceptAccordion';
import {
  DEFAULT_CONFIG,
  deterministicPolicy,
  type Policy,
} from '@/lib/rl/gridworld';
import {
  generateNormalSamples,
  batchVsIncrementalMean,
  robbinsMonroSequence,
  powerStepSizes,
  checkStepSizeCondition,
  meanEstimationGradientDescent,
  tdBridgeStep,
  type RMFunction,
  type GDMode,
} from '@/lib/rl/stochasticApproximation';

const RIGHT_POLICY: (0 | 1 | 2 | 3 | 4)[] = [1, 1, 1, 1, 1, 1, 1, 1, 1];

type TabKey = 'mean' | 'rm' | 'stepsize' | 'gd' | 'td';

export default function Chapter06SaPage() {
  const [activeTab, setActiveTab] = useState<TabKey>('mean');

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <TrendingDown className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          第 6 章 随机逼近
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          从增量均值到 Robbins-Monro、Dvoretzky 定理与随机梯度下降，为第 7 章的 TD 方法奠定数学基础。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="Robbins-Monro 算法"
          formula={
            <KaTeX
              math={String.raw`w_{k+1} = w_k - \alpha_k \tilde{g}(w_k, \eta_k)`}
              display
            />
          }
          description="用带噪声的观测 g̃ 迭代求方程 g(w)=0 的根。"
        />
        <FormulaCard
          title="Dvoretzky 收敛条件"
          formula={
            <KaTeX
              math={String.raw`\mathbb{E}[w_{k+1} - w^* \mid \mathcal{H}_k] \le (1-\alpha_k)(w_k-w^*) + \beta_k, \quad \sum \alpha_k=\infty, \sum \beta_k<\infty`}
              display
            />
          }
          description="更一般的随机逼近收敛框架，Robbins-Monro 与均值估计都是其特例。"
        />
        <FormulaCard
          title="随机梯度下降"
          formula={
            <KaTeX
              math={String.raw`w_{k+1} = w_k - \alpha_k \widetilde{\nabla} J(w_k)`}
              display
            />
          }
          description="用单个或小批量样本近似批量梯度，是函数近似与深度 RL 的优化基础。"
        />
      </section>

      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as TabKey)}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="mean">批量 vs 增量均值</TabsTrigger>
          <TabsTrigger value="rm">Robbins-Monro</TabsTrigger>
          <TabsTrigger value="stepsize">步长条件</TabsTrigger>
          <TabsTrigger value="gd">BGD/MBGD/SGD</TabsTrigger>
          <TabsTrigger value="td">SA → TD</TabsTrigger>
        </TabsList>

        <TabsContent value="mean" className="mt-4">
          <BatchIncrementalMeanDemo />
        </TabsContent>
        <TabsContent value="rm" className="mt-4">
          <RobbinsMonroDemo />
        </TabsContent>
        <TabsContent value="stepsize" className="mt-4">
          <StepSizeConditionDemo />
        </TabsContent>
        <TabsContent value="gd" className="mt-4">
          <GDDemo />
        </TabsContent>
        <TabsContent value="td" className="mt-4">
          <TDBridgeDemo />
        </TabsContent>
      </Tabs>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">小结与 Q&A</h2>
        <ConceptAccordion
          items={[
            {
              id: 'summary',
              title: '本章小结',
              content: (
                <ul className="list-disc pl-5 space-y-2">
                  <li>增量均值更新是随机逼近的最简单形式：w_{'{k+1}'} = w_k + α_k(x_k - w_k)。</li>
                  <li>Robbins-Monro 用带噪声观测求方程 g(w)=0 的根。</li>
                  <li>经典收敛要求 Σα_k=∞ 且 Σα_k²&lt;∞；固定步长不满足几乎处处收敛，但适合非平稳跟踪。</li>
                  <li>Dvoretzky 定理给出更一般的随机递推收敛条件，RM 与 SGD 均可视为特例。</li>
                  <li>BGD、MBGD、SGD 在梯度方差与每次计算成本之间做不同权衡。</li>
                  <li>TD(0) 把随机逼近中的“随机样本”替换为“随机 TD target”，是第 6 章通往第 7 章的桥梁。</li>
                </ul>
              ),
            },
            {
              id: 'qa1',
              title: 'Q: 为什么需要 Σα_k = ∞ 且 Σα_k² < ∞？',
              content:
                'Σα_k = ∞ 保证算法能走足够远以到达目标；Σα_k² < ∞ 保证噪声累积的方差有限，从而收敛。',
            },
          ]}
        />
      </section>
    </div>
  );
}

// ------------------- Tab 1: Batch vs Incremental Mean -------------------
function BatchIncrementalMeanDemo() {
  const [seed, setSeed] = useState(1);
  const [trueMean, setTrueMean] = useState(2);
  const [std, setStd] = useState(1);
  const [totalSamples, setTotalSamples] = useState(50);
  const [currentIndex, setCurrentIndex] = useState(0);

  const samples = useMemo(
    () => generateNormalSamples(totalSamples, trueMean, std, seed),
    [totalSamples, trueMean, std, seed]
  );
  const history = useMemo(() => batchVsIncrementalMean(samples), [samples]);
  const current = history[currentIndex - 1];

  useEffect(() => {
    setCurrentIndex(0);
  }, [seed, trueMean, std, totalSamples]);

  function next() {
    if (currentIndex < totalSamples) setCurrentIndex((i) => i + 1);
  }
  function reset() {
    setCurrentIndex(0);
  }

  return (
    <InteractiveDemo title="从批量均值到增量均值">
      <p className="text-xs text-gray-500 mb-3">
        同一组样本同时用批量公式和增量公式更新。每步只到一个新样本；批量方法需保存全部样本，增量方法只保存当前估计。
        这就是从第 5 章非增量方法通向第 7 章增量 TD 的第一步。
      </p>
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <LineChart
              data={history.slice(0, currentIndex).map((h, i) => ({
                step: i + 1,
                batch: h.batchMean,
                incremental: h.incrementalMean,
                trueMean,
              }))}
              xKey="step"
              xLabel="样本数 n"
              yLabel="均值估计"
              series={[
                { key: 'trueMean', name: '真实均值', color: '#22c55e' },
                { key: 'batch', name: '批量均值', color: '#2563eb' },
                { key: 'incremental', name: '增量均值', color: '#ef4444' },
              ]}
              height={240}
            />
          </div>
          {current && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">当前样本 x_n</div><div className="font-mono">{current.sample.toFixed(3)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">旧估计</div><div className="font-mono">{(current.incrementalMean - current.stepSize * current.predictionError).toFixed(3)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">prediction error</div><div className="font-mono">{current.predictionError.toFixed(3)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">步长 1/n</div><div className="font-mono">{current.stepSize.toFixed(4)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">更新后估计</div><div className="font-mono font-semibold">{current.incrementalMean.toFixed(3)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">批量结果</div><div className="font-mono">{current.batchMean.toFixed(3)}</div></CardContent></Card>
              <Card><CardContent className="p-3 col-span-2"><div className="text-gray-500 text-xs">内存差异</div><div>批量保存 {current.n} 个数；增量只保存 1 个估计</div></CardContent></Card>
            </div>
          )}
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">参数</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <Param label="真实均值" value={trueMean} set={setTrueMean} min={-5} max={5} step={0.5} />
              <Param label="标准差" value={std} set={setStd} min={0.1} max={2} step={0.1} fixed={1} />
              <Param label="总样本数" value={totalSamples} set={setTotalSamples} min={10} max={200} step={10} />
            </CardContent>
          </Card>
          <div className="flex gap-2">
            <Button onClick={next} disabled={currentIndex >= totalSamples} className="flex-1">下一步</Button>
            <Button onClick={reset} variant="outline" className="flex-1">重置</Button>
          </div>
          <Button onClick={() => { setSeed((s) => s + 1); setCurrentIndex(0); }} variant="outline" className="w-full">重新生成样本</Button>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- Tab 2: Robbins-Monro -------------------
function RobbinsMonroDemo() {
  const [seed, setSeed] = useState(1);
  const [wStar, setWStar] = useState(3);
  const [initialW, setInitialW] = useState(0);
  const [power, setPower] = useState(0.75);
  const [steps, setSteps] = useState(100);
  const [noiseStd, setNoiseStd] = useState(0.5);
  const [gName, setGName] = useState<RMFunction>('linear');
  const [stepIndex, setStepIndex] = useState(0);

  const alphas = useMemo(() => powerStepSizes(steps, power), [steps, power]);
  const history = useMemo(
    () => robbinsMonroSequence(wStar, initialW, alphas, noiseStd, gName, seed),
    [wStar, initialW, alphas, noiseStd, gName, seed]
  );

  useEffect(() => {
    setStepIndex(0);
  }, [seed, wStar, initialW, power, steps, noiseStd, gName]);

  const current = history[stepIndex];
  const chartData = useMemo(
    () =>
      history.slice(0, stepIndex + 1).map((h, i) => ({
        step: i,
        w: h.w,
        wStar,
      })),
    [history, stepIndex, wStar]
  );

  return (
    <InteractiveDemo title="Robbins-Monro：带噪声观测求根">
      <p className="text-xs text-gray-500 mb-3">
        只能观测 g̃(w_k,η_k)=g(w_k)+η_k，用 w_{'{k+1}'}=w_k-α_k g̃ 迭代求 g(w*)=0。
      </p>
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <LineChart
              data={chartData}
              xKey="step"
              xLabel="迭代步 k"
              yLabel="w_k"
              series={[
                { key: 'wStar', name: '真实根 w*', color: '#22c55e' },
                { key: 'w', name: 'w_k', color: '#2563eb' },
              ]}
              height={240}
            />
          </div>
          {current && (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">当前 w_k</div><div className="font-mono">{current.w.toFixed(3)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">真实 g(w_k)</div><div className="font-mono">{current.trueG.toFixed(3)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">带噪声观测</div><div className="font-mono">{current.noisyG.toFixed(3)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">更新量</div><div className="font-mono">{current.update.toFixed(4)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">步长 α_k</div><div className="font-mono">{current.alpha.toFixed(4)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">w_{'{k+1}'}</div><div className="font-mono font-semibold">{current.wNext.toFixed(3)}</div></CardContent></Card>
            </div>
          )}
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">参数</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <Param label="真实根 w*" value={wStar} set={setWStar} min={-5} max={5} step={0.5} />
              <Param label="初始 w₀" value={initialW} set={setInitialW} min={-5} max={5} step={0.5} />
              <Param label="步长幂 p (α_k=1/k^p)" value={power} set={setPower} min={0.3} max={1.3} step={0.05} fixed={2} />
              <Param label="迭代次数" value={steps} set={setSteps} min={20} max={300} step={10} />
              <Param label="噪声标准差" value={noiseStd} set={setNoiseStd} min={0} max={2} step={0.1} fixed={1} />
              <div>
                <label className="text-sm text-gray-700 block mb-1">函数 g(w)</label>
                <Select value={gName} onValueChange={(v) => setGName(v as RMFunction)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="linear">g(w)=w-w*</SelectItem>
                    <SelectItem value="tanh">g(w)=tanh(w-w*)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>
          <div className="flex gap-2">
            <Button onClick={() => setStepIndex((i) => Math.min(i + 1, history.length - 1))} disabled={stepIndex >= history.length - 1} className="flex-1">下一步</Button>
            <Button onClick={() => setStepIndex(0)} variant="outline" className="flex-1">重置</Button>
          </div>
          <Button onClick={() => { setSeed((s) => s + 1); setStepIndex(0); }} variant="outline" className="w-full">重新生成噪声</Button>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- Tab 3: Step Size Condition Checker -------------------
const PRESET_POWERS = [0.3, 0.5, 0.75, 1.0, 1.25];

function StepSizeConditionDemo() {
  const [p, setP] = useState(0.75);
  const condition = useMemo(() => checkStepSizeCondition(p), [p]);

  return (
    <InteractiveDemo title="步长条件检查器：α_k = 1/k^p">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <div className="text-sm text-gray-700 space-y-2">
              <p>
                经典 Robbins-Monro 条件要求：
                <KaTeX math={String.raw`\sum \alpha_k = \infty, \quad \sum \alpha_k^2 < \infty`} display />
              </p>
              <p className="font-semibold text-lg">
                当前 p = <span className="font-mono">{p.toFixed(2)}</span>
              </p>
              <p className={`font-semibold ${condition.valid ? 'text-green-600' : 'text-amber-600'}`}>
                {condition.label} — {condition.valid ? '满足经典条件' : '不满足经典条件'}
              </p>
              <p className="text-gray-600">{condition.description}</p>
            </div>
          </div>
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200 text-sm text-gray-700 space-y-2">
            <p><strong>固定步长</strong>不满足经典几乎处处收敛条件，但可在根附近形成稳态波动，适合追踪非平稳目标。</p>
          </div>
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">参数</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <Param label="幂 p" value={p} set={setP} min={0.1} max={1.5} step={0.05} fixed={2} />
              <div className="flex flex-wrap gap-2">
                {PRESET_POWERS.map((pv) => (
                  <Button key={pv} size="sm" variant="outline" onClick={() => setP(pv)}>
                    p={pv}
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- Tab 4: BGD / MBGD / SGD -------------------
function GDDemo() {
  const [seed, setSeed] = useState(1);
  const [trueMean, setTrueMean] = useState(2);
  const [std, setStd] = useState(1);
  const [n, setN] = useState(100);
  const [initialW, setInitialW] = useState(0);
  const [alpha, setAlpha] = useState(0.1);
  const [epochs, setEpochs] = useState(20);
  const [mode, setMode] = useState<GDMode>('sgd');
  const [batchSize, setBatchSize] = useState(1);

  const dataset = useMemo(() => generateNormalSamples(n, trueMean, std, seed), [n, trueMean, std, seed]);
  const history = useMemo(
    () => meanEstimationGradientDescent(dataset, initialW, mode, batchSize, alpha, epochs, seed),
    [dataset, initialW, mode, batchSize, alpha, epochs, seed]
  );

  const chartData = useMemo(
    () =>
      history.map((h) => ({
        step: h.step,
        w: h.w,
        loss: h.loss,
        trueMean,
      })),
    [history, trueMean]
  );

  const final = history[history.length - 1];

  return (
    <InteractiveDemo title="BGD / Mini-batch GD / SGD：同一数据集上的比较">
      <p className="text-xs text-gray-500 mb-3">
        目标 J(w)=1/(2n)Σ(w-x_i)²。BGD 用全部样本，MBGD 用 m 个样本，SGD 用 1 个样本。
        改变算法时数据集与噪声序列保持不变（同一 seed），可公平比较。
      </p>
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <LineChart
              data={chartData}
              xKey="step"
              xLabel="更新步"
              yLabel="w_k / loss"
              series={[
                { key: 'trueMean', name: '真实均值', color: '#22c55e' },
                { key: 'w', name: '参数 w', color: '#2563eb' },
                { key: 'loss', name: 'loss', color: '#ef4444' },
              ]}
              height={240}
            />
          </div>
          {final && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">最终 w</div><div className="font-mono">{final.w.toFixed(3)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">最终 loss</div><div className="font-mono">{final.loss.toFixed(4)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">每步 batch size</div><div className="font-mono">{final.batchSize}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">累计样本数</div><div className="font-mono">{final.samplesProcessed}</div></CardContent></Card>
            </div>
          )}
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">参数</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <Param label="真实均值" value={trueMean} set={setTrueMean} min={-5} max={5} step={0.5} />
              <Param label="标准差" value={std} set={setStd} min={0.1} max={2} step={0.1} fixed={1} />
              <Param label="样本数 n" value={n} set={setN} min={20} max={200} step={10} />
              <Param label="初始 w₀" value={initialW} set={setInitialW} min={-5} max={5} step={0.5} />
              <Param label="学习率 α" value={alpha} set={setAlpha} min={0.001} max={0.5} step={0.001} fixed={3} />
              <Param label="迭代轮数" value={epochs} set={setEpochs} min={1} max={50} step={1} />
              <div>
                <label className="text-sm text-gray-700 block mb-1">更新模式</label>
                <Select value={mode} onValueChange={(v) => setMode(v as GDMode)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="bgd">BGD（全部样本）</SelectItem>
                    <SelectItem value="mbgd">Mini-batch GD</SelectItem>
                    <SelectItem value="sgd">SGD（单样本）</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {mode === 'mbgd' && (
                <div>
                  <label className="text-sm text-gray-700 block mb-1">Mini-batch size m</label>
                  <Select value={String(batchSize)} onValueChange={(v) => setBatchSize(Number(v))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {[1, 2, 5, 10, 20, n].map((m) => (
                        <SelectItem key={m} value={String(m)}>m={m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
            </CardContent>
          </Card>
          <Button onClick={() => setSeed((s) => s + 1)} variant="outline" className="w-full">重新生成样本</Button>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- Tab 5: SA → TD Bridge -------------------
function TDBridgeDemo() {
  const config = DEFAULT_CONFIG;
  const policy: Policy = useMemo(() => deterministicPolicy(RIGHT_POLICY, 5), []);
  const [seed, setSeed] = useState(1);
  const [alpha, setAlpha] = useState(0.2);
  const [v, setV] = useState<number[]>(() => new Array(config.rows * config.cols).fill(0));
  const [history, setHistory] = useState<ReturnType<typeof tdBridgeStep>['step'][]>([]);

  const stateValues = v;

  function step() {
    const { step: s, vNew } = tdBridgeStep(config, policy, v, alpha, seed + history.length);
    setV(vNew);
    setHistory((prev) => [...prev, s]);
  }

  function reset() {
    setV(new Array(config.rows * config.cols).fill(0));
    setHistory([]);
  }

  const last = history[history.length - 1];

  return (
    <InteractiveDemo title="从随机逼近到 TD(0)">
      <p className="text-xs text-gray-500 mb-3">
        均值估计 w_{'{k+1}'}=w_k+α_k(x_k-w_k) 与 TD(0) 更新 V(S_t)←V(S_t)+α[R_{'{t+1}'}+γV(S_{'{t+1}'})-V(S_t)] 具有相同结构：
        w_k ↔ V(S_t)，随机样本 x_k ↔ 随机 TD target，prediction error ↔ TD error，E[X] ↔ Bellman fixed point。
      </p>
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld config={config} policy={policy} values={stateValues} showValues className="max-w-full" />
            <p className="mt-2 text-xs text-gray-500">当前值函数 V(s)（在被评估策略 π 下）</p>
          </div>
          {last && (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">状态 S_t</div><div className="font-mono">s{last.state + 1}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">奖励 R_{'{t+1}'}</div><div className="font-mono">{last.reward.toFixed(2)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">下一状态 S_{'{t+1}'}</div><div className="font-mono">s{last.nextState + 1}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">旧 V(S_t)</div><div className="font-mono">{last.vOld.toFixed(3)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">TD target</div><div className="font-mono">{last.tdTarget.toFixed(3)}</div></CardContent></Card>
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">TD error δ</div><div className="font-mono font-semibold">{last.tdError.toFixed(3)}</div></CardContent></Card>
            </div>
          )}
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">参数</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <Param label="学习率 α" value={alpha} set={setAlpha} min={0.01} max={1} step={0.01} fixed={2} />
            </CardContent>
          </Card>
          <div className="flex gap-2">
            <Button onClick={step} className="flex-1">走一步 TD 更新</Button>
            <Button onClick={reset} variant="outline" className="flex-1">重置 V</Button>
          </div>
          <Button onClick={() => { setSeed((s) => s + 1); reset(); }} variant="outline" className="w-full">重新生成样本序列</Button>
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">对应关系</CardTitle></CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-1">
              <div>w_k ↔ V(S_t)</div>
              <div>x_k ↔ R_{'{t+1}'} + γV(S_{'{t+1}'})</div>
              <div>x_k - w_k ↔ TD error δ_t</div>
              <div>E[X] ↔ Bellman fixed point</div>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- Shared Param -------------------
function Param({
  label,
  value,
  set,
  min,
  max,
  step,
  fixed,
}: {
  label: string;
  value: number;
  set: (v: number) => void;
  min: number;
  max: number;
  step: number;
  fixed?: number;
}) {
  return (
    <div>
      <div className="text-sm text-gray-600 mb-1">{label}</div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={([v]) => set(v)} />
      <div className="mt-1 text-center font-mono text-sm text-gray-700">
        {fixed !== undefined ? value.toFixed(fixed) : value}
      </div>
    </div>
  );
}
