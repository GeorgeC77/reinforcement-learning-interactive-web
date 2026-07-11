import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import {
  TrendingDown,
  ShieldAlert,
  Play,
  Pause,
  RotateCcw,
  StepForward,
} from 'lucide-react';
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
  randomPolicy,
  type Policy,
  type Action,
  type GridWorldConfig,
  isTerminal,
} from '@/lib/rl/gridworld';
import {
  mulberry32,
  generateNormalSamples,
  batchVsIncrementalMean,
  robbinsMonroSequence,
  evaluateGPrime,
  powerStepSizes,
  checkStepSizeCondition,
  partialSumCondition,
  meanEstimationGradientDescent,
  estimateBatchGradientVariance,
  movingAverage,
  tdBridgeStep,
  dvoretzkyErrorSequence,
  type RMFunction,
  type GDMode,
  type GDStep,
} from '@/lib/rl/stochasticApproximation';

// Default 3x3 GridWorld policies for the TD bridge demo.
const GOAL_POLICY: Action[] = [2, 2, 3, 1, 2, 3, 0, 1, 4];
const RIGHT_POLICY: Action[] = [1, 1, 1, 1, 1, 1, 1, 1, 1];
const CUSTOM_INITIAL: Action[] = [...RIGHT_POLICY];

type TabKey = 'mean' | 'rm' | 'stepsize' | 'gd' | 'td' | 'dvoretzky';
type GDViewMode = 'single' | 'compare';
type GDXAxis = 'step' | 'epoch' | 'samplesProcessed';
type TaskType = 'continuing' | 'episodic';
type PolicyPreset = 'goal' | 'random' | 'custom';

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
          从增量均值到 Robbins-Monro、Dvoretzky 收敛定理与随机梯度下降，
          为第 7 章的 TD 方法奠定数学基础。
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
          title="Dvoretzky 收敛定理"
          formula={
            <KaTeX
              math={String.raw`\Delta_{k+1} = (1-\alpha_k)\Delta_k + \beta_k \eta_k, \quad \Delta_k \xrightarrow{a.s.} 0`}
              display
            />
          }
          description="统一的随机逼近收敛框架；Robbins-Monro 与均值估计都是其特例。"
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
        <TabsList className="flex w-full overflow-x-auto">
          <TabsTrigger value="mean" className="whitespace-nowrap">批量 vs 增量均值</TabsTrigger>
          <TabsTrigger value="rm" className="whitespace-nowrap">Robbins-Monro</TabsTrigger>
          <TabsTrigger value="stepsize" className="whitespace-nowrap">步长条件</TabsTrigger>
          <TabsTrigger value="gd" className="whitespace-nowrap">BGD/MBGD/SGD</TabsTrigger>
          <TabsTrigger value="td" className="whitespace-nowrap">SA → TD</TabsTrigger>
          <TabsTrigger value="dvoretzky" className="whitespace-nowrap">Dvoretzky 定理</TabsTrigger>
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
        <TabsContent value="dvoretzky" className="mt-4">
          <DvoretzkyDemo />
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
                  <li>Robbins-Monro 用带噪声观测求方程 g(w)=0 的根；收敛需要函数、步长、噪声三方面条件。</li>
                  <li>经典步长条件 Σα_k=∞ 且 Σα_k²&lt;∞ 只是定理的一部分。</li>
                  <li>Dvoretzky 定理给出更一般的随机递推收敛条件，RM 与 SGD 均可视为特例。</li>
                  <li>BGD、MBGD、SGD 用 full objective 比较才公平；梯度噪声用 batch gradient 与 full gradient 之差衡量。</li>
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

  const gPrime = current ? evaluateGPrime(current.w, wStar, gName) : null;
  const stepCondition = useMemo(() => checkStepSizeCondition(power), [power]);
  const partial = useMemo(() => partialSumCondition(steps, power), [steps, power]);

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
              <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">g'(w_k)</div><div className="font-mono">{gPrime?.toFixed(3)}</div></CardContent></Card>
            </div>
          )}

          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">Robbins-Monro 收敛条件检查器</CardTitle></CardHeader>
            <CardContent className="text-sm space-y-3">
              <ConditionRow
                title="1. 函数条件"
                ok={gName === 'linear'}
                okText="g(w)=w-w* 满足 g(w*)=0，严格单调，且 g'(w)≡1∈[c1,c2]。"
                warnText={
                  <span>
                    当前 g(w)=tanh(w-w*) 满足 g(w*)=0 且单调，但其导数 g&apos;(x)=sech²(x) 在全局没有统一正下界，
                    不能直接套用教材中要求 0&lt;c1≤g&apos;≤c2 的全局 RM 定理。该示例主要用于直观演示，实际收敛还与初值和局部性质有关。
                  </span>
                }
              />
              <ConditionRow
                title="2. 步长条件"
                ok={stepCondition.valid}
                okText={`当前 p=${power.toFixed(2)}：Σα_k 发散且 Σα_k² 收敛，满足经典步长条件（部分和 Σα=${partial.sum.toFixed(2)}，Σα²=${partial.sumSquares.toFixed(3)}）。`}
                warnText={`当前 p=${power.toFixed(2)} 不满足经典步长条件。步长条件只是 RM 定理的一部分，单独满足它不足以保证收敛。`}
              />
              <ConditionRow
                title="3. 噪声条件"
                ok
                okText={`观测噪声 η_k~N(0,${noiseStd}²)，满足 E[η_k|H_k]=0 且二阶矩有限。`}
              />
              <p className="text-amber-700 bg-amber-50 p-2 rounded">
                提示：步长条件只是 Robbins-Monro 定理的一部分，单独满足步长条件并不足以保证收敛；函数条件与噪声条件同样不可或缺。
              </p>
            </CardContent>
          </Card>
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

function ConditionRow({
  title,
  ok,
  okText,
  warnText,
}: {
  title: string;
  ok: boolean;
  okText: React.ReactNode;
  warnText?: React.ReactNode;
}) {
  return (
    <div className="flex gap-3">
      <div className={`mt-0.5 w-2 h-2 rounded-full flex-shrink-0 ${ok ? 'bg-green-500' : 'bg-amber-500'}`} />
      <div>
        <div className="font-semibold text-gray-800">{title}</div>
        <div className={ok ? 'text-green-700' : 'text-amber-700'}>{ok ? okText : warnText}</div>
      </div>
    </div>
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
  const [singleMode, setSingleMode] = useState<GDMode>('sgd');
  const [mbBatchSize, setMbBatchSize] = useState(10);
  const [viewMode, setViewMode] = useState<GDViewMode>('single');
  const [xAxis, setXAxis] = useState<GDXAxis>('samplesProcessed');

  const dataset = useMemo(() => generateNormalSamples(n, trueMean, std, seed), [n, trueMean, std, seed]);

  const singleHistory = useMemo(
    () => meanEstimationGradientDescent(dataset, initialW, singleMode, mbBatchSize, alpha, epochs, seed),
    [dataset, initialW, singleMode, mbBatchSize, alpha, epochs, seed]
  );

  const compareHistories = useMemo(
    () => ({
      bgd: meanEstimationGradientDescent(dataset, initialW, 'bgd', n, alpha, epochs, seed),
      mbgd: meanEstimationGradientDescent(dataset, initialW, 'mbgd', mbBatchSize, alpha, epochs, seed),
      sgd: meanEstimationGradientDescent(dataset, initialW, 'sgd', 1, alpha, epochs, seed),
    }),
    [dataset, initialW, mbBatchSize, alpha, epochs, seed]
  );

  const noiseWindow = 10;

  const singleChartData = useMemo(
    () =>
      singleHistory.map((h) => ({
        [xAxis]: xAxis === 'step' ? h.step : xAxis === 'epoch' ? h.epoch : h.samplesProcessed,
        w: h.wAfter,
        loss: h.fullLossAfter,
        trueMean,
      })),
    [singleHistory, xAxis, trueMean]
  );

  const compareChartData = useMemo(
    () => buildCompareChartData(compareHistories, xAxis, noiseWindow, trueMean),
    [compareHistories, xAxis, noiseWindow, trueMean]
  );

  const fixedWNoise = useMemo(() => {
    return {
      mbgd: estimateBatchGradientVariance(dataset, initialW, 'mbgd', mbBatchSize, 200, seed),
      sgd: estimateBatchGradientVariance(dataset, initialW, 'sgd', 1, 200, seed),
    };
  }, [dataset, initialW, mbBatchSize, seed]);

  const finalSingle = singleHistory[singleHistory.length - 1];

  const xLabel =
    xAxis === 'step' ? '更新步' : xAxis === 'epoch' ? 'epoch' : '累计样本数';

  return (
    <InteractiveDemo title="BGD / Mini-batch GD / SGD：公平比较">
      <p className="text-xs text-gray-500 mb-3">
        目标 J(w)=1/(2n)Σ(w-x_i)²。比较模式使用同一 dataset、同一 seed、同一初始 w、同一学习率、同一总样本预算（每轮 epoch 遍历全部 n 个样本一次）。
        主 loss 曲线使用 full objective 在 wAfter 处取值，保证 w 与 loss 属于同一时点。
      </p>
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          {viewMode === 'single' ? (
            <>
              <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
                <LineChart
                  data={singleChartData}
                  xKey={xAxis}
                  xLabel={xLabel}
                  yLabel="w / loss"
                  series={[
                    { key: 'trueMean', name: '真实均值', color: '#22c55e' },
                    { key: 'w', name: 'w', color: '#2563eb' },
                    { key: 'loss', name: 'full loss', color: '#ef4444' },
                  ]}
                  height={240}
                />
              </div>
              {finalSingle && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                  <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">最终 w</div><div className="font-mono">{finalSingle.wAfter.toFixed(3)}</div></CardContent></Card>
                  <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">最终 full loss</div><div className="font-mono">{finalSingle.fullLossAfter.toFixed(4)}</div></CardContent></Card>
                  <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">batch size</div><div className="font-mono">{finalSingle.batchSize}</div></CardContent></Card>
                  <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">累计样本</div><div className="font-mono">{finalSingle.samplesProcessed}</div></CardContent></Card>
                </div>
              )}
            </>
          ) : (
            <>
              <MetricChart title="参数 w" data={compareChartData} xKey={xAxis} xLabel={xLabel} yLabel="w" metric="w" trueMean={trueMean} />
              <MetricChart title="Full Objective Loss" data={compareChartData} xKey={xAxis} xLabel={xLabel} yLabel="loss" metric="loss" />
              <MetricChart title="Squared Gradient Noise 移动平均" data={compareChartData} xKey={xAxis} xLabel={xLabel} yLabel="squared noise" metric="noise" />
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                {(['bgd', 'mbgd', 'sgd'] as GDMode[]).map((mode) => {
                  const h = compareHistories[mode];
                  const final = h[h.length - 1];
                  return (
                    <Card key={mode}>
                      <CardHeader className="pb-2"><CardTitle className="text-base">{mode.toUpperCase()} 最终</CardTitle></CardHeader>
                      <CardContent className="space-y-1">
                        <div className="flex justify-between"><span>w</span><span className="font-mono">{final.wAfter.toFixed(3)}</span></div>
                        <div className="flex justify-between"><span>full loss</span><span className="font-mono">{final.fullLossAfter.toFixed(4)}</span></div>
                        <div className="flex justify-between"><span>gradient noise²</span><span className="font-mono">{final.squaredGradientNoise.toExponential(2)}</span></div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </>
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
              <Param label="epoch 数" value={epochs} set={setEpochs} min={1} max={50} step={1} />
              <div>
                <label className="text-sm text-gray-700 block mb-1">视图</label>
                <Select value={viewMode} onValueChange={(v) => setViewMode(v as GDViewMode)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="single">单算法分析</SelectItem>
                    <SelectItem value="compare">三算法同图比较</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {viewMode === 'single' && (
                <div>
                  <label className="text-sm text-gray-700 block mb-1">算法</label>
                  <Select value={singleMode} onValueChange={(v) => setSingleMode(v as GDMode)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="bgd">BGD</SelectItem>
                      <SelectItem value="mbgd">Mini-batch GD</SelectItem>
                      <SelectItem value="sgd">SGD</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}
              {(viewMode === 'compare' || singleMode !== 'bgd') && (
                <div>
                  <label className="text-sm text-gray-700 block mb-1">Mini-batch size m</label>
                  <Select value={String(mbBatchSize)} onValueChange={(v) => setMbBatchSize(Number(v))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {[1, 2, 5, 10, 20, n].map((m) => (
                        <SelectItem key={m} value={String(m)}>m={m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
              <div>
                <label className="text-sm text-gray-700 block mb-1">横轴</label>
                <Select value={xAxis} onValueChange={(v) => setXAxis(v as GDXAxis)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="samplesProcessed">累计样本数</SelectItem>
                    <SelectItem value="step">更新步</SelectItem>
                    <SelectItem value="epoch">epoch</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">固定 w 下的梯度噪声估计</CardTitle></CardHeader>
            <CardContent className="text-sm space-y-2">
              <p className="text-gray-600">在 w₀={initialW} 处重复抽取 200 个 batch 估计 Var[g_B(w)]：</p>
              <div className="grid grid-cols-2 gap-3">
                <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">MBGD Var[g_B]</div><div className="font-mono">{fixedWNoise.mbgd.varianceOfBatchGradients.toExponential(2)}</div></CardContent></Card>
                <Card><CardContent className="p-3"><div className="text-gray-500 text-xs">SGD Var[g_B]</div><div className="font-mono">{fixedWNoise.sgd.varianceOfBatchGradients.toExponential(2)}</div></CardContent></Card>
              </div>
              <p className="text-xs text-gray-500">BGD 的 batch 就是全集，梯度噪声为 0。</p>
            </CardContent>
          </Card>

          <Button onClick={() => setSeed((s) => s + 1)} variant="outline" className="w-full">重新生成样本</Button>
        </div>
      </div>
    </InteractiveDemo>
  );
}

function buildCompareChartData(
  histories: Record<GDMode, GDStep[]>,
  xAxis: GDXAxis,
  noiseWindow: number,
  trueMean: number
) {
  const dataMap = new Map<number, Record<string, number>>();
  const xKey = xAxis === 'step' ? 'step' : xAxis === 'epoch' ? 'epoch' : 'samples';

  const noiseMA: Record<GDMode, number[]> = {
    bgd: movingAverage(histories.bgd.map((h) => h.squaredGradientNoise), noiseWindow),
    mbgd: movingAverage(histories.mbgd.map((h) => h.squaredGradientNoise), noiseWindow),
    sgd: movingAverage(histories.sgd.map((h) => h.squaredGradientNoise), noiseWindow),
  };

  (Object.keys(histories) as GDMode[]).forEach((mode) => {
    histories[mode].forEach((h, idx) => {
      const x = xAxis === 'step' ? h.step : xAxis === 'epoch' ? h.epoch : h.samplesProcessed;
      if (!dataMap.has(x)) dataMap.set(x, { [xKey]: x, trueMean });
      const obj = dataMap.get(x)!;
      obj[`w_${mode}`] = h.wAfter;
      obj[`loss_${mode}`] = h.fullLossAfter;
      obj[`noise_${mode}`] = noiseMA[mode][idx];
    });
  });

  return Array.from(dataMap.values()).sort((a, b) => (a[xKey] as number) - (b[xKey] as number));
}

function MetricChart({
  title,
  data,
  xKey,
  xLabel,
  yLabel,
  metric,
  trueMean,
}: {
  title: string;
  data: Record<string, number>[];
  xKey: string;
  xLabel: string;
  yLabel: string;
  metric: string;
  trueMean?: number;
}) {
  const series = [
    { key: `${metric}_bgd`, name: 'BGD', color: '#2563eb' },
    { key: `${metric}_mbgd`, name: 'MBGD', color: '#f59e0b' },
    { key: `${metric}_sgd`, name: 'SGD', color: '#ef4444' },
  ];
  if (metric === 'w' && trueMean !== undefined) {
    series.push({ key: 'trueMean', name: '真实均值', color: '#22c55e' });
  }
  return (
    <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
      <h4 className="text-sm font-semibold text-gray-700 mb-2">{title}</h4>
      <LineChart data={data} xKey={xKey} xLabel={xLabel} yLabel={yLabel} series={series} height={220} />
    </div>
  );
}

// ------------------- Tab 5: SA → TD Bridge -------------------
function TDBridgeDemo() {
  const [taskType, setTaskType] = useState<TaskType>('continuing');
  const [policyPreset, setPolicyPreset] = useState<PolicyPreset>('goal');
  const [customActions, setCustomActions] = useState<Action[]>([...CUSTOM_INITIAL]);
  const [alpha, setAlpha] = useState(0.2);
  const [seed, setSeed] = useState(1);
  const [v, setV] = useState<number[]>(() => new Array(DEFAULT_CONFIG.rows * DEFAULT_CONFIG.cols).fill(0));
  const [currentState, setCurrentState] = useState(DEFAULT_CONFIG.startState);
  const [stepCount, setStepCount] = useState(0);
  const [history, setHistory] = useState<ReturnType<typeof tdBridgeStep>['step'][]>([]);
  const [visitedStates, setVisitedStates] = useState<number[]>([DEFAULT_CONFIG.startState]);
  const [isRunning, setIsRunning] = useState(false);

  const config: GridWorldConfig = useMemo(
    () => ({ ...DEFAULT_CONFIG, taskType }),
    [taskType]
  );
  const policy: Policy = useMemo(
    () =>
      policyPreset === 'random'
        ? randomPolicy(config.rows * config.cols, 5)
        : deterministicPolicy(policyPreset === 'goal' ? GOAL_POLICY : customActions, 5),
    [config, policyPreset, customActions]
  );

  const stateRef = useRef({ v, currentState, stepCount, history, visitedStates });
  useEffect(() => {
    stateRef.current = { v, currentState, stepCount, history, visitedStates };
  }, [v, currentState, stepCount, history, visitedStates]);

  const runStep = useCallback(() => {
    const { v: curV, currentState: curState, stepCount: curStep, history: curHistory, visitedStates: curVisited } = stateRef.current;
    let state = curState;
    if (isTerminal(state, config)) state = config.startState;
    const rng = mulberry32(seed + curStep);
    const res = tdBridgeStep(config, policy, curV, alpha, state, rng);
    setV(res.vNew);
    setCurrentState(res.nextCurrentState);
    setStepCount(curStep + 1);
    setHistory([...curHistory, res.step]);
    setVisitedStates([...curVisited, res.nextCurrentState]);
  }, [config, policy, alpha, seed]);

  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(runStep, 250);
    return () => clearInterval(id);
  }, [isRunning, runStep]);

  function resetEpisode() {
    setCurrentState(config.startState);
    setVisitedStates([config.startState]);
  }
  function resetValues() {
    setV(new Array(config.rows * config.cols).fill(0));
    setHistory([]);
    setStepCount(0);
    setIsRunning(false);
  }
  function resetAll() {
    resetEpisode();
    resetValues();
  }

  function handleTaskTypeChange(t: TaskType) {
    setTaskType(t);
    resetAll();
  }
  function handlePresetChange(p: PolicyPreset) {
    setPolicyPreset(p);
    resetEpisode();
  }

  const last = history[history.length - 1];

  return (
    <InteractiveDemo title="从随机逼近到 TD(0)：连续交互轨迹">
      <p className="text-xs text-gray-500 mb-3">
        均值估计 w_{'{k+1}'}=w_k+α_k(x_k-w_k) 与 TD(0) 更新 V(S_t)←V(S_t)+α[R_{'{t+1}'}+γV(S_{'{t+1}'})-V(S_t)] 具有相同结构：
        w_k ↔ V(S_t)，随机样本 x_k ↔ 随机 TD target，prediction error ↔ TD error，E[X] ↔ Bellman fixed point。
      </p>
      <div className="grid lg:grid-cols-[1fr_360px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld
              config={config}
              policy={policy}
              values={v}
              showValues
              highlightState={currentState}
              trajectory={visitedStates}
              currentStep={visitedStates.length - 1}
              editable={policyPreset === 'custom'}
              onActionClick={(state, action) => {
                setCustomActions((prev) => {
                  const next = [...prev];
                  next[state] = action as Action;
                  return next;
                });
              }}
              className="max-w-full"
            />
            <p className="mt-2 text-xs text-gray-500">当前值函数 V(s) 与 agent 位置（蓝框）</p>
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
          {visitedStates.length > 1 && (
            <div className="text-sm text-gray-700 bg-gray-50 rounded-xl p-4 border border-gray-200">
              <strong>已走轨迹：</strong> {visitedStates.map((s) => `s${s + 1}`).join(' → ')}
            </div>
          )}
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">参数</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <div>
                <label className="text-sm text-gray-700 block mb-1">任务类型</label>
                <Select value={taskType} onValueChange={(v) => handleTaskTypeChange(v as TaskType)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="continuing">Continuing（教材默认）</SelectItem>
                    <SelectItem value="episodic">Episodic（寻路）</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="text-sm text-gray-700 block mb-1">策略预设</label>
                <Select value={policyPreset} onValueChange={(v) => handlePresetChange(v as PolicyPreset)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="goal">通向目标（默认）</SelectItem>
                    <SelectItem value="random">Uniform Random</SelectItem>
                    <SelectItem value="custom">用户自定义（点击箭头）</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Param label="学习率 α" value={alpha} set={setAlpha} min={0.01} max={1} step={0.01} fixed={2} />
            </CardContent>
          </Card>

          <div className="flex flex-wrap gap-2">
            <Button onClick={runStep} className="flex-1 min-w-[80px]"><StepForward className="w-4 h-4 mr-1" />单步</Button>
            <Button onClick={() => setIsRunning((r) => !r)} variant={isRunning ? 'default' : 'outline'} className="flex-1 min-w-[80px]">
              {isRunning ? <><Pause className="w-4 h-4 mr-1" />暂停</> : <><Play className="w-4 h-4 mr-1" />自动</>}
            </Button>
            <Button onClick={resetEpisode} variant="outline" className="flex-1 min-w-[80px]"><RotateCcw className="w-4 h-4 mr-1" />重置回合</Button>
            <Button onClick={resetValues} variant="outline" className="flex-1 min-w-[80px]"><RotateCcw className="w-4 h-4 mr-1" />重置 V</Button>
          </div>

          <Button onClick={() => { setSeed((s) => s + 1); resetAll(); }} variant="outline" className="w-full">重新生成随机序列</Button>

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

// ------------------- Tab 6: Dvoretzky Theorem -------------------
function DvoretzkyDemo() {
  const [alphaPower, setAlphaPower] = useState(0.75);
  const [betaPower, setBetaPower] = useState(0.75);
  const [steps, setSteps] = useState(100);
  const [initialDelta, setInitialDelta] = useState(1);
  const [noiseStd, setNoiseStd] = useState(0.5);
  const [seed, setSeed] = useState(1);
  const [showProof, setShowProof] = useState(false);

  const alphas = useMemo(() => powerStepSizes(steps, alphaPower), [steps, alphaPower]);
  const betas = useMemo(() => powerStepSizes(steps, betaPower), [steps, betaPower]);
  const history = useMemo(
    () => dvoretzkyErrorSequence(initialDelta, alphas, betas, noiseStd, seed),
    [initialDelta, alphas, betas, noiseStd, seed]
  );

  const alphaCondition = useMemo(() => checkStepSizeCondition(alphaPower), [alphaPower]);
  const betaCondition = useMemo(() => checkStepSizeCondition(betaPower), [betaPower]);
  const alphaPartial = useMemo(() => partialSumCondition(steps, alphaPower), [steps, alphaPower]);
  const betaPartial = useMemo(() => partialSumCondition(steps, betaPower), [steps, betaPower]);

  const chartData = useMemo(
    () => history.map((h) => ({ k: h.k, delta: h.delta })),
    [history]
  );

  return (
    <InteractiveDemo title="Dvoretzky 收敛定理">
      <div className="grid lg:grid-cols-[1fr_360px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200 text-sm text-gray-700 space-y-3">
            <p className="font-semibold">Theorem 6.2（Dvoretzky&apos;s convergence theorem）</p>
            <p>考虑随机过程</p>
            <KaTeX math={String.raw`\Delta_{k+1} = (1-\alpha_k)\Delta_k + \beta_k \eta_k, \quad k=1,2,\dots`} display />
            <p>
              其中 {String.raw`\{\alpha_k\}, \{\beta_k\}, \{\eta_k\}`} 为随机序列，且 {String.raw`\alpha_k\ge0, \beta_k\ge0`}。
              若满足：
            </p>
            <ol className="list-decimal pl-5 space-y-1">
              <li>
                系数/步长条件（一致地 a.s.）：
                <KaTeX math={String.raw`\sum_{k=1}^\infty \alpha_k = \infty, \quad \sum_{k=1}^\infty \alpha_k^2 < \infty, \quad \sum_{k=1}^\infty \beta_k^2 < \infty`} display />
              </li>
              <li>
                噪声条件：
                <KaTeX math={String.raw`\mathbb{E}[\eta_k \mid \mathcal{H}_k] = 0, \quad \mathbb{E}[\eta_k^2 \mid \mathcal{H}_k] \le C`} display />
                其中 {String.raw`\mathcal{H}_k = \{\Delta_k,\Delta_{k-1},\dots,\eta_{k-1},\dots,\alpha_{k-1},\dots,\beta_{k-1},\dots\}`} 为历史 filtration。
              </li>
            </ol>
            <p>则</p>
            <KaTeX math={String.raw`\Delta_k \xrightarrow{a.s.} 0, \quad k\to\infty`} display />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <ConditionCheckCard
              title="系数 α_k 条件"
              ok={alphaCondition.valid}
              details={`p_α=${alphaPower.toFixed(2)}：Σα=${alphaPartial.sum.toFixed(2)}，Σα²=${alphaPartial.sumSquares.toFixed(3)}。${alphaCondition.description}`}
            />
            <ConditionCheckCard
              title="系数 β_k 条件"
              ok={betaCondition.valid}
              details={`p_β=${betaPower.toFixed(2)}：Σβ=${betaPartial.sum.toFixed(2)}，Σβ²=${betaPartial.sumSquares.toFixed(3)}。${betaCondition.description}`}
            />
            <ConditionCheckCard
              title="条件均值 E[η_k|H_k]=0"
              ok
              details="演示噪声为零均值高斯噪声，条件均值为 0。"
            />
            <ConditionCheckCard
              title="条件方差 E[η_k²|H_k]≤C"
              ok
              details={`噪声标准差固定为 ${noiseStd}，条件二阶矩有界。`}
            />
            <ConditionCheckCard
              title="结论"
              ok={alphaCondition.valid && betaCondition.valid}
              details={alphaCondition.valid && betaCondition.valid ? 'α 与 β 均满足经典条件，Δ_k → 0 almost surely。' : 'α 或 β 不满足经典条件，无法保证 almost sure 收敛。'}
            />
          </div>

          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <h4 className="text-sm font-semibold text-gray-700 mb-2">误差递推 Δ_k 演示</h4>
            <LineChart data={chartData} xKey="k" xLabel="k" yLabel="Δ_k" series={[{ key: 'delta', name: 'Δ_k', color: '#2563eb' }]} height={220} />
          </div>

          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">应用于均值估计</CardTitle></CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <KaTeX math={String.raw`w_{k+1} = w_k + \alpha_k (x_k - w_k)`} display />
              <p>令 {String.raw`\Delta_k = w_k - \mathbb{E}[X]`}，则</p>
              <KaTeX math={String.raw`\Delta_{k+1} = (1-\alpha_k)\Delta_k + \alpha_k (x_k - \mathbb{E}[X])`} display />
              <p>这里 {String.raw`\alpha_k`} 既是收缩系数也是 β_k，而 {String.raw`\eta_k = x_k - \mathbb{E}[X]`} 满足零均值、有限方差，从而直接落入 Dvoretzky 框架。</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">应用于 Robbins-Monro</CardTitle></CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <KaTeX math={String.raw`w_{k+1} = w_k - \alpha_k (g(w_k) + \eta_k)`} display />
              <p>设 w* 为根，利用中值定理：</p>
              <KaTeX math={String.raw`g(w_k) - g(w^*) = \nabla_w g(w_k') (w_k - w^*)`} display />
              <p>令 {String.raw`\Delta_k = w_k - w^*`}，可得</p>
              <KaTeX math={String.raw`\Delta_{k+1} = \bigl(1 - \alpha_k \nabla_w g(w_k')\bigr)\Delta_k + \alpha_k (-\eta_k)`} display />
              <p>
                当 {String.raw`0 < c_1 \le \nabla_w g(w) \le c_2`} 且步长满足条件时，上式即 Dvoretzky 形式，故 {String.raw`w_k \xrightarrow{a.s.} w^*`}。
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">扩展到多状态：Q-learning 收敛分析</CardTitle></CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <p>对有限状态/动作集合，可定义逐状态误差 {String.raw`\Delta_k(s) = Q_k(s) - Q^*(s)`}，得到</p>
              <KaTeX math={String.raw`\Delta_{k+1}(s) = (1-\alpha_k(s))\Delta_k(s) + \beta_k(s) \eta_k(s)`} display />
              <p>
                扩展 Dvoretzky 定理要求：对每个 s，Σα_k(s)=∞、Σα_k²(s)&lt;∞、Σβ_k²(s)&lt;∞，
                并且通常要求 {String.raw`\mathbb{E}[\beta_k(s) \mid \mathcal{H}_k] \le \mathbb{E}[\alpha_k(s) \mid \mathcal{H}_k]`}。
                在最大范数下可证明所有状态的误差同时趋于 0，从而得到 Q-learning 的 almost sure 收敛。
              </p>
            </CardContent>
          </Card>

          <Button onClick={() => setShowProof((s) => !s)} variant="outline" className="w-full">
            {showProof ? '隐藏证明思路' : '展开证明思路'}
          </Button>
          {showProof && (
            <div className="bg-gray-50 rounded-xl p-6 border border-gray-200 text-sm text-gray-700 space-y-2">
              <p>证明核心：将递推两边取条件期望，得到 Lyapunov 函数 {String.raw`\mathbb{E}[\Delta_k^2]`} 的递推不等式</p>
              <KaTeX math={String.raw`\mathbb{E}[\Delta_{k+1}^2 \mid \mathcal{H}_k] \le (1-\alpha_k)^2 \Delta_k^2 + \beta_k^2 C`} display />
              <p>
                利用 Σα=∞ 保证足够更新，Σα²&lt;∞ 与 Σβ²&lt;∞ 控制噪声积累，
                结合 martingale 收敛定理可推出 {String.raw`\Delta_k \xrightarrow{a.s.} 0`}。
                详细证明参见教材 Theorem 6.2。
              </p>
            </div>
          )}
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">参数</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <Param label="初始误差 Δ₁" value={initialDelta} set={setInitialDelta} min={-5} max={5} step={0.5} />
              <Param label="α 幂 p_α" value={alphaPower} set={setAlphaPower} min={0.3} max={1.3} step={0.05} fixed={2} />
              <Param label="β 幂 p_β" value={betaPower} set={setBetaPower} min={0.3} max={1.3} step={0.05} fixed={2} />
              <Param label="迭代步数" value={steps} set={setSteps} min={20} max={300} step={10} />
              <Param label="噪声标准差" value={noiseStd} set={setNoiseStd} min={0} max={2} step={0.1} fixed={1} />
            </CardContent>
          </Card>
          <Button onClick={() => setSeed((s) => s + 1)} variant="outline" className="w-full">重新生成噪声</Button>
        </div>
      </div>
    </InteractiveDemo>
  );
}

function ConditionCheckCard({ title, ok, details }: { title: string; ok: boolean; details: React.ReactNode }) {
  return (
    <Card className={ok ? 'border-green-200' : 'border-amber-200'}>
      <CardContent className="p-3 text-sm">
        <div className="flex items-center gap-2 mb-1">
          <div className={`w-2 h-2 rounded-full ${ok ? 'bg-green-500' : 'bg-amber-500'}`} />
          <span className="font-semibold text-gray-800">{title}</span>
        </div>
        <div className={ok ? 'text-green-700' : 'text-amber-700'}>{details}</div>
      </CardContent>
    </Card>
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
