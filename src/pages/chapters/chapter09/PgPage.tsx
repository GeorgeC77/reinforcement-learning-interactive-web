import { useState, useMemo } from 'react';
import { GitBranch, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import LineChart from '@/components/LineChart';
import ConceptAccordion from '@/components/ConceptAccordion';
import { reinforceBandit, reinforceWithBaseline } from '@/lib/rl/gridworld';

export default function Chapter09PgPage() {
  const [tab, setTab] = useState('reinforce');
  const [seed, setSeed] = useState(0);

  const [alpha, setAlpha] = useState(0.2);
  const [beta, setBeta] = useState(0.1);
  const [episodes, setEpisodes] = useState(200);

  const actionRewards = [1, 3];
  const initialTheta = [0, 0];

  const reinforceData = useMemo(() => {
    void seed;
    const { policyHistory, rewardHistory } = reinforceBandit(
      actionRewards,
      initialTheta,
      alpha,
      episodes
    );
    return policyHistory.map((policy, i) => ({
      episode: i,
      action0: policy[0],
      action1: policy[1],
      reward: rewardHistory[i - 1] ?? 0,
    }));
  }, [alpha, episodes, seed]);

  const baselineData = useMemo(() => {
    void seed;
    const { policyHistory, rewardHistory, baselineHistory } = reinforceWithBaseline(
      actionRewards,
      initialTheta,
      alpha,
      beta,
      episodes
    );
    return policyHistory.map((policy, i) => ({
      episode: i,
      action0: policy[0],
      action1: policy[1],
      baseline: baselineHistory[i],
      reward: rewardHistory[i - 1] ?? 0,
    }));
  }, [alpha, beta, episodes, seed]);

  const finalPolicy = reinforceData[reinforceData.length - 1];

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <GitBranch className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          第 9 章 策略梯度方法
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          策略梯度定理、REINFORCE 与带基线的 REINFORCE：直接参数化并优化策略。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="策略梯度定理"
          formula={
            <KaTeX
              math={String.raw`\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot q_{\pi_\theta}(s,a) \right]`}
              display
            />
          }
          description="策略目标对参数的梯度可写成 log-策略梯度与动作值的期望。"
        />
        <FormulaCard
          title="Softmax 策略"
          formula={<KaTeX math={String.raw`\pi(a) = \frac{e^{\theta_a}}{\sum_{a'} e^{\theta_{a'}}}`} display />}
          description="动作偏好 θ 越大，选择该动作的概率越高。"
        />
        <FormulaCard
          title="带基线的 REINFORCE"
          formula={
            <KaTeX
              math={String.raw`\theta_a \leftarrow \theta_a + \alpha (G - b) \bigl( \mathbb{I}(a) - \pi(a) \bigr)`}
              display
            />
          }
          description="用基线 b（如平均回报）降低方差，不改变期望梯度。"
        />
      </section>

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="reinforce">REINFORCE</TabsTrigger>
          <TabsTrigger value="baseline">REINFORCE + Baseline</TabsTrigger>
        </TabsList>

        <TabsContent value="reinforce" className="mt-4">
          <InteractiveDemo title="REINFORCE 训练过程">
            <div className="grid lg:grid-cols-[1fr_340px] gap-6">
              <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
                <LineChart
                  data={reinforceData}
                  xKey="episode"
                  xLabel="回合"
                  yLabel="选择概率"
                  series={[
                    { key: 'action0', name: '动作 0 (r≈1)', color: '#2563eb' },
                    { key: 'action1', name: '动作 1 (r≈3)', color: '#ef4444' },
                  ]}
                />
              </div>
              <div className="space-y-4">
                <ParamCard>
                  <Param label="α" value={alpha} set={setAlpha} min={0.01} max={0.5} step={0.01} fixed={2} />
                  <Param label="回合数" value={episodes} set={setEpisodes} min={20} max={500} step={10} />
                </ParamCard>
                <StrategyCard policy={finalPolicy} />
                <Button onClick={() => setSeed((s) => s + 1)} variant="outline" className="w-full">重新采样</Button>
              </div>
            </div>
          </InteractiveDemo>
        </TabsContent>

        <TabsContent value="baseline" className="mt-4">
          <InteractiveDemo title="带基线的 REINFORCE：降低方差">
            <div className="grid lg:grid-cols-[1fr_340px] gap-6">
              <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
                <LineChart
                  data={baselineData}
                  xKey="episode"
                  xLabel="回合"
                  yLabel="概率 / 基线"
                  series={[
                    { key: 'action0', name: '动作 0', color: '#2563eb' },
                    { key: 'action1', name: '动作 1', color: '#ef4444' },
                    { key: 'baseline', name: '基线 b', color: '#22c55e', strokeDasharray: '4 2' },
                  ]}
                />
              </div>
              <div className="space-y-4">
                <ParamCard>
                  <Param label="α" value={alpha} set={setAlpha} min={0.01} max={0.5} step={0.01} fixed={2} />
                  <Param label="β（基线更新）" value={beta} set={setBeta} min={0.001} max={0.5} step={0.001} fixed={3} />
                  <Param label="回合数" value={episodes} set={setEpisodes} min={20} max={500} step={10} />
                </ParamCard>
                <Card>
                  <CardHeader className="pb-2"><CardTitle className="text-base">基线作用</CardTitle></CardHeader>
                  <CardContent className="text-sm text-gray-700 space-y-2">
                    <p>回报 G 大于基线时，提升该动作概率；小于基线时，降低概率。</p>
                    <p>基线只改变梯度估计的方差，不改变期望，因此策略梯度定理仍然成立。</p>
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
                  <li>策略梯度直接优化策略参数，适用于连续或大动作空间。</li>
                  <li>策略梯度定理把目标梯度转化为 log-策略梯度与动作值的期望。</li>
                  <li>REINFORCE 用蒙特卡洛回报估计梯度，方差大。</li>
                  <li>引入基线可显著降低方差，且不影响梯度期望。</li>
                </ul>
              ),
            },
            {
              id: 'qa1',
              title: 'Q: 为什么基线能降低方差但不改变期望？',
              content:
                '因为 Σ_a ∇log π(a) π(a) b = b ∇Σ_a π(a) = b ∇1 = 0，所以减去基线后期望梯度不变，但估计量的方差减小。',
            },
          ]}
        />
      </section>
    </div>
  );
}

function ParamCard({ children }: { children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader className="pb-2"><CardTitle className="text-base">参数</CardTitle></CardHeader>
      <CardContent className="space-y-3">{children}</CardContent>
    </Card>
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

function StrategyCard({ policy }: { policy: { action0: number; action1: number } }) {
  return (
    <Card>
      <CardHeader className="pb-2"><CardTitle className="text-base">当前策略</CardTitle></CardHeader>
      <CardContent className="space-y-2 text-sm text-gray-700">
        <div>动作 0 概率：<span className="font-mono">{policy.action0.toFixed(3)}</span></div>
        <div>动作 1 概率：<span className="font-mono">{policy.action1.toFixed(3)}</span></div>
      </CardContent>
    </Card>
  );
}
