import { useState, useMemo } from 'react';
import { TrendingDown, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import LineChart from '@/components/LineChart';
import ConceptAccordion from '@/components/ConceptAccordion';
import {
  sgdQuadratic,
  robbinsMonro,
  dvoretzkyMeanEstimation,
  powerStepSizes,
} from '@/lib/rl/gridworld';

export default function Chapter06SaPage() {
  const [tab, setTab] = useState('sgd');
  const [seed, setSeed] = useState(0);

  // SGD params
  const [wTrue, setWTrue] = useState(3);
  const [initialW, setInitialW] = useState(0);
  const [alpha, setAlpha] = useState(0.05);
  const [noiseStd, setNoiseStd] = useState(0.5);
  const [iterations, setIterations] = useState(100);

  // RM / Dvoretzky params
  const [rmTrue, setRmTrue] = useState(3);
  const [rmInit, setRmInit] = useState(0);
  const [rmSteps, setRmSteps] = useState(100);
  const [power, setPower] = useState(1);

  const sgdData = useMemo(() => {
    void seed;
    const hist = sgdQuadratic(wTrue, initialW, alpha, noiseStd, iterations);
    return hist.map((w, i) => ({ iteration: i, value: w }));
  }, [wTrue, initialW, alpha, noiseStd, iterations, seed]);

  const rmData = useMemo(() => {
    void seed;
    const steps = powerStepSizes(rmSteps, power);
    const rm = robbinsMonro(rmTrue, rmInit, steps);
    const dv = dvoretzkyMeanEstimation(rmTrue, rmInit, steps);
    return rm.map((w, i) => ({ iteration: i, robbinsMonro: w, dvoretzky: dv[i] }));
  }, [rmTrue, rmInit, rmSteps, power, seed]);

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
          Robbins-Monro、Dvoretzky 定理与随机梯度下降：为 TD、Q-learning 等增量算法奠定数学基础。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="Robbins-Monro 算法"
          formula={<KaTeX math={String.raw`w_{k+1} = w_k - \alpha_k \tilde{g}(w_k, \eta_k)`} display />}
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
          description="更一般的随机逼近收敛框架，Robbins-Monro 是其特例。"
        />
        <FormulaCard
          title="随机梯度下降"
          formula={<KaTeX math={String.raw`w_{k+1} = w_k - \alpha_k \widetilde{\nabla} J(w_k)`} display />}
          description="用单个样本近似批量梯度，是函数近似与深度 RL 的优化基础。"
        />
      </section>

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="sgd">SGD</TabsTrigger>
          <TabsTrigger value="rm">Robbins-Monro</TabsTrigger>
          <TabsTrigger value="dv">Dvoretzky</TabsTrigger>
        </TabsList>

        <TabsContent value="sgd" className="mt-4">
          <InteractiveDemo title="SGD 优化二次函数">
            <div className="grid lg:grid-cols-[1fr_340px] gap-6">
              <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
                <LineChart
                  data={sgdData}
                  xKey="iteration"
                  xLabel="迭代次数"
                  yLabel="w"
                  series={[{ key: 'value', name: 'w_k', color: '#2563eb' }]}
                />
              </div>
              <div className="space-y-4">
                <Card>
                  <CardHeader className="pb-2"><CardTitle className="text-base">参数</CardTitle></CardHeader>
                  <CardContent className="space-y-3">
                    <Param label="w*" value={wTrue} set={setWTrue} min={-5} max={5} step={0.5} />
                    <Param label="w₀" value={initialW} set={setInitialW} min={-5} max={5} step={0.5} />
                    <Param label="α" value={alpha} set={setAlpha} min={0.001} max={0.2} step={0.001} fixed={3} />
                    <Param label="噪声" value={noiseStd} set={setNoiseStd} min={0} max={2} step={0.1} fixed={1} />
                    <Param label="迭代" value={iterations} set={setIterations} min={20} max={300} step={10} />
                  </CardContent>
                </Card>
                <Button onClick={() => setSeed((s) => s + 1)} variant="outline" className="w-full">重新采样</Button>
              </div>
            </div>
          </InteractiveDemo>
        </TabsContent>

        <TabsContent value="rm" className="mt-4">
          <InteractiveDemo title="Robbins-Monro：不同步长序列">
            <div className="grid lg:grid-cols-[1fr_340px] gap-6">
              <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
                <LineChart
                  data={rmData}
                  xKey="iteration"
                  xLabel="迭代次数"
                  yLabel="w"
                  series={[{ key: 'robbinsMonro', name: 'Robbins-Monro', color: '#2563eb' }]}
                />
              </div>
              <div className="space-y-4">
                <Card>
                  <CardHeader className="pb-2"><CardTitle className="text-base">参数</CardTitle></CardHeader>
                  <CardContent className="space-y-3">
                    <Param label="w*" value={rmTrue} set={setRmTrue} min={-5} max={5} step={0.5} />
                    <Param label="w₀" value={rmInit} set={setRmInit} min={-5} max={5} step={0.5} />
                    <Param label="步长 α_k = 1/k^p" value={power} set={setPower} min={0.3} max={1.5} step={0.1} fixed={1} />
                    <Param label="迭代" value={rmSteps} set={setRmSteps} min={20} max={300} step={10} />
                  </CardContent>
                </Card>
                <Button onClick={() => setSeed((s) => s + 1)} variant="outline" className="w-full">重新采样</Button>
              </div>
            </div>
          </InteractiveDemo>
        </TabsContent>

        <TabsContent value="dv" className="mt-4">
          <InteractiveDemo title="Dvoretzky vs Robbins-Monro 均值估计">
            <div className="grid lg:grid-cols-[1fr_340px] gap-6">
              <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
                <LineChart
                  data={rmData}
                  xKey="iteration"
                  xLabel="迭代次数"
                  yLabel="估计值"
                  series={[
                    { key: 'robbinsMonro', name: 'Robbins-Monro', color: '#2563eb' },
                    { key: 'dvoretzky', name: 'Dvoretzky SA', color: '#ef4444' },
                  ]}
                />
              </div>
              <div className="space-y-4">
                <Card>
                  <CardHeader className="pb-2"><CardTitle className="text-base">说明</CardTitle></CardHeader>
                  <CardContent className="text-sm text-gray-700 space-y-2">
                    <p>Robbins-Monro 用 <KaTeX math={String.raw`w - \alpha \tilde{g}(w)`} /> 更新。</p>
                    <p>Dvoretzky 形式用 <KaTeX math={String.raw`w + \alpha (x - w)`} /> 更新，直接估计均值。</p>
                    <p>二者在均值估计问题上是等价的，都依赖 <KaTeX math={String.raw`\sum \alpha_k = \infty, \sum \alpha_k^2 < \infty`} />。</p>
                  </CardContent>
                </Card>
                <Button onClick={() => setSeed((s) => s + 1)} variant="outline" className="w-full">重新采样</Button>
              </div>
            </div>
          </InteractiveDemo>
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
                  <li>随机逼近是用带噪声样本迭代求解方程或优化问题的方法族。</li>
                  <li>Robbins-Monro 和 Dvoretzky 定理给出了收敛条件。</li>
                  <li>SGD 是随机逼近在优化问题中的代表。</li>
                  <li>TD、Sarsa、Q-learning 都可以视为随机逼近求解 Bellman/BOE。</li>
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
