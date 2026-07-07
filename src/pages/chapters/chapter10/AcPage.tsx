import { useState, useMemo, useEffect } from 'react';
import { Activity, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import AlgorithmPlayer from '@/components/AlgorithmPlayer';
import LineChart from '@/components/LineChart';
import ConceptAccordion from '@/components/ConceptAccordion';
import {
  EPISODIC_PATH_CONFIG,
  qac,
  a2c,
  offPolicyActorCritic,
  actionValueToStateValue,
} from '@/lib/rl/gridworld';

type TabKey = 'qac' | 'a2c' | 'offpolicy' | 'deterministic';

export default function Chapter10AcPage() {
  const [tab, setTab] = useState<TabKey>('qac');

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Activity className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          第 10 章 Actor-Critic 方法
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          QAC、A2C、异策略 Actor-Critic 与确定性 Actor-Critic：用 Critic 估计值或动作值，引导 Actor 更新。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="QAC：Q-function Actor-Critic"
          formula={
            <KaTeX
              math={String.raw`\theta \leftarrow \theta + \alpha_\theta q(s,a,w) \nabla_\theta \log \pi_\theta(a|s)`}
              display
            />
          }
          description="Critic 直接估计 q(s,a)，Actor 用 q(s,a) 作为权重更新策略。"
        />
        <FormulaCard
          title="A2C：Advantage Actor-Critic"
          formula={
            <KaTeX
              math={String.raw`\delta_t = r + \gamma v(s') - v(s), \quad \theta \leftarrow \theta + \alpha_\theta \delta_t \nabla_\theta \log \pi_\theta(a|s)`}
              display
            />
          }
          description="Advantage 用 TD error 近似，Critic 估计状态值 V(s)。"
        />
        <FormulaCard
          title="异策略 Actor-Critic"
          formula={
            <KaTeX
              math={String.raw`\rho_t = \frac{\pi_\theta(a_t|s_t)}{\beta(a_t|s_t)}, \quad \theta \leftarrow \theta + \alpha_\theta \rho_t q(s_t,a_t) \nabla_\theta \log \pi_\theta(a_t|s_t)`}
              display
            />
          }
          description="行为策略 β 采样，目标策略 π 更新，重要性采样比 ρ 修正分布差异。"
        />
        <FormulaCard
          title="确定性 Actor-Critic"
          formula={
            <KaTeX
              math={String.raw`\nabla_\theta J \approx \nabla_a Q(s,a,w)\big|_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s)`}
              display
            />
          }
          description="策略确定性地输出动作；Actor 沿 Critic 对动作的梯度方向调整。"
        />
      </section>

      <Tabs value={tab} onValueChange={(v) => setTab(v as TabKey)}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="qac">QAC</TabsTrigger>
          <TabsTrigger value="a2c">A2C</TabsTrigger>
          <TabsTrigger value="offpolicy">异策略 AC</TabsTrigger>
          <TabsTrigger value="deterministic">确定性 AC</TabsTrigger>
        </TabsList>

        <TabsContent value="qac" className="mt-4">
          <DiscreteACDemo algorithm="qac" />
        </TabsContent>
        <TabsContent value="a2c" className="mt-4">
          <DiscreteACDemo algorithm="a2c" />
        </TabsContent>
        <TabsContent value="offpolicy" className="mt-4">
          <DiscreteACDemo algorithm="offpolicy" />
        </TabsContent>
        <TabsContent value="deterministic" className="mt-4">
          <DeterministicACDemo />
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
                  <li>Actor-Critic 同时学习策略（Actor）和值函数（Critic）。</li>
                  <li>QAC 用 Critic 估计 q(s,a) 直接驱动 Actor。</li>
                  <li>A2C 用 TD error 作为 advantage 估计，降低方差。</li>
                  <li>异策略 AC 用重要性采样比 ρ 把行为策略样本转换为目标策略梯度。</li>
                  <li>确定性 AC 适用于连续动作，Actor 沿 Critic 对动作的梯度方向移动。</li>
                </ul>
              ),
            },
            {
              id: 'qa1',
              title: 'Q: “异策略”和“离线强化学习”是一回事吗？',
              content:
                '不是。异策略（off-policy）指训练时用的行为策略与要优化的目标策略不同，但仍是在线交互学习；离线强化学习（offline RL）指完全使用预先收集好的固定数据集，不再与环境交互。这里演示的是异策略 AC。',
            },
            {
              id: 'qa2',
              title: 'Q: A2C 和 QAC 的 Critic 有什么区别？',
              content:
                'A2C 的 Critic 估计状态值 V(s)，advantage 用 TD error 近似；QAC 的 Critic 估计动作值 q(s,a)，直接用它作为 Actor 的权重。',
            },
          ]}
        />
      </section>
    </div>
  );
}

// ------------------- Discrete AC tabs (QAC / A2C / Off-policy) -------------------
function DiscreteACDemo({ algorithm }: { algorithm: 'qac' | 'a2c' | 'offpolicy' }) {
  const config = EPISODIC_PATH_CONFIG;
  const [actorAlpha, setActorAlpha] = useState(0.05);
  const [criticAlpha, setCriticAlpha] = useState(0.1);
  const [epsilon, setEpsilon] = useState(algorithm === 'offpolicy' ? 0.5 : 0.3);
  const [episodes, setEpisodes] = useState(150);
  const [step, setStep] = useState(0);

  useEffect(() => {
    setStep(0);
  }, [algorithm, actorAlpha, criticAlpha, epsilon, episodes]);

  const history = useMemo(() => {
    if (algorithm === 'qac') {
      const { qHistory, policies, rewardHistory } = qac(config, actorAlpha, criticAlpha, episodes);
      return qHistory.map((q, i) => ({
        values: actionValueToStateValue(q),
        policy: policies[i],
        reward: rewardHistory[i],
      }));
    }
    if (algorithm === 'a2c') {
      const { values, policies, rewardHistory } = a2c(config, actorAlpha, criticAlpha, episodes);
      return values.map((v, i) => ({ values: v, policy: policies[i], reward: rewardHistory[i] }));
    }
    const { qHistory, policies, rewardHistory } = offPolicyActorCritic(
      config,
      actorAlpha,
      criticAlpha,
      epsilon,
      episodes
    );
    return qHistory.map((q, i) => ({
      values: actionValueToStateValue(q),
      policy: policies[i],
      reward: rewardHistory[i],
    }));
  }, [algorithm, config, actorAlpha, criticAlpha, epsilon, episodes]);

  const current = history[Math.min(step, history.length - 1)];
  const maxStep = history.length - 1;

  const rewardData = useMemo(
    () =>
      history.slice(1).map((h, i) => ({
        episode: i + 1,
        reward: h.reward,
      })),
    [history]
  );

  const title =
    algorithm === 'qac'
      ? 'QAC：Critic 估计 q(s,a) 驱动 Actor'
      : algorithm === 'a2c'
      ? 'A2C：用 TD error 作为 advantage'
      : '异策略 Actor-Critic：用 ρ 修正行为策略样本';

  return (
    <InteractiveDemo title={title}>
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld
              config={config}
              policy={current.policy}
              values={current.values}
              showValues
              className="max-w-full"
            />
            <p className="mt-3 text-sm text-gray-500 text-center">第 {step} 回合后的策略与值函数</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <LineChart
              data={rewardData}
              xKey="episode"
              xLabel="回合"
              yLabel="回合总回报"
              series={[{ key: 'reward', name: 'Episode Return', color: '#2563eb' }]}
              height={200}
            />
          </div>
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">超参数</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <Param label="Actor α" value={actorAlpha} set={setActorAlpha} min={0.001} max={0.2} step={0.001} fixed={3} />
              <Param label="Critic α" value={criticAlpha} set={setCriticAlpha} min={0.001} max={0.5} step={0.001} fixed={3} />
              {algorithm === 'offpolicy' && (
                <Param label="行为策略 ε" value={epsilon} set={setEpsilon} min={0.05} max={1} step={0.05} fixed={2} />
              )}
              <Param label="训练回合数" value={episodes} set={setEpisodes} min={20} max={300} step={10} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">回放控制</CardTitle></CardHeader>
            <CardContent>
              <AlgorithmPlayer maxStep={maxStep} currentStep={step} onStepChange={setStep} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">环境说明</CardTitle></CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <p>使用 episodic path task：到达目标即终止，每步 -1，禁区 -10，撞边界 -10。</p>
              <p>回报曲线应随训练逐渐上升（路径越来越短）。</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- Deterministic AC Demo -------------------
function DeterministicACDemo() {
  const [actorTheta0, setActorTheta0] = useState(0.5);
  const [actorTheta1, setActorTheta1] = useState(0.0);
  const [criticW, setCriticW] = useState<number[]>([0, 0, 0, 0, 0, 0]);
  const [episode, setEpisode] = useState(0);

  const actorAlpha = 0.05;
  const criticAlpha = 0.1;
  const gamma = 0.9;

  function features(s: number, a: number): number[] {
    return [1, s, a, s * s, s * a, a * a];
  }

  function qValue(s: number, a: number): number {
    return features(s, a).reduce((sum, f, i) => sum + f * criticW[i], 0);
  }

  function mu(s: number): number {
    const x = actorTheta0 * s + actorTheta1;
    return Math.tanh(x);
  }

  function gradMu(s: number): number[] {
    const m = mu(s);
    const factor = 1 - m * m;
    return [factor * s, factor];
  }

  function gradQWrtA(s: number, a: number): number {
    // derivative of w^T phi w.r.t a
    return criticW[2] + criticW[4] * s + 2 * criticW[5] * a;
  }

  function runEpisode() {
    let s = -0.8;
    let totalReward = 0;
    const steps = 20;
    let newW = [...criticW];
    let newTheta0 = actorTheta0;
    let newTheta1 = actorTheta1;

    for (let t = 0; t < steps; t++) {
      const a = mu(s);
      const nextS = Math.max(-1, Math.min(1, s + 0.3 * a));
      const r = -(nextS * nextS);
      totalReward += r;

      const nextA = mu(nextS);
      const target = r + gamma * qValue(nextS, nextA);
      const pred = qValue(s, a);
      const tdError = target - pred;

      // Critic update
      const phi = features(s, a);
      for (let i = 0; i < newW.length; i++) {
        newW[i] += criticAlpha * tdError * phi[i];
      }

      // Actor update: dJ/dtheta = dQ/da * dmu/dtheta
      const dqda = gradQWrtA(s, a);
      const [dmu0, dmu1] = gradMu(s);
      newTheta0 += actorAlpha * dqda * dmu0;
      newTheta1 += actorAlpha * dqda * dmu1;

      s = nextS;
      if (Math.abs(s) < 0.05) break;
    }

    setCriticW(newW);
    setActorTheta0(newTheta0);
    setActorTheta1(newTheta1);
    setEpisode((e) => e + 1);
  }

  function reset() {
    setActorTheta0(0.5);
    setActorTheta1(0.0);
    setCriticW([0, 0, 0, 0, 0, 0]);
    setEpisode(0);
  }

  const stateForPlot = 0.3;
  const actionForPlot = mu(stateForPlot);
  const plotActions = Array.from({ length: 41 }, (_, i) => -1 + (2 * i) / 40);
  const plotValues = plotActions.map((a) => qValue(stateForPlot, a));
  const maxQ = Math.max(...plotValues);
  const minQ = Math.min(...plotValues);
  const range = maxQ - minQ || 1;

  return (
    <InteractiveDemo title="确定性 Actor-Critic：一维连续动作示意">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
          <div className="text-sm text-gray-600 mb-2">固定状态 s = {stateForPlot.toFixed(1)} 下的 Critic Q(s,a)</div>
          <svg width={360} height={220} className="bg-white rounded-lg border border-gray-200">
            {plotActions.map((a, i) => {
              const x = 30 + ((a + 1) / 2) * 300;
              const y = 190 - ((plotValues[i] - minQ) / range) * 160;
              return (
                <circle key={i} cx={x} cy={y} r={2} fill="#2563eb" />
              );
            })}
            {/* Actor action marker */}
            <circle
              cx={30 + ((actionForPlot + 1) / 2) * 300}
              cy={190 - ((qValue(stateForPlot, actionForPlot) - minQ) / range) * 160}
              r={6}
              fill="#ef4444"
            />
            <text x={180} y={210} textAnchor="middle" fontSize={10} fill="#6b7280">动作 a</text>
            <text x={10} y={15} fontSize={10} fill="#6b7280">Q</text>
          </svg>
          <p className="mt-3 text-sm text-gray-500 text-center">
            蓝点为 Q(s,a) 曲线，红点为当前确定性策略选择的动作 μ(s)。Actor 沿 Q 增大方向调整 μ。
          </p>
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">训练统计</CardTitle></CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <div>已训练回合：<span className="font-mono font-semibold">{episode}</span></div>
              <div>Actor μ(s) = tanh(θ₀·s + θ₁)</div>
              <div>θ₀ = <span className="font-mono">{actorTheta0.toFixed(3)}</span></div>
              <div>θ₁ = <span className="font-mono">{actorTheta1.toFixed(3)}</span></div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">说明</CardTitle></CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <p>状态 s∈[-1,1]，动作 a∈[-1,1]，目标是把状态推向 0。</p>
              <p>Critic 用线性 Q(s,a) 近似；Actor 用 tanh 输出确定性动作。</p>
              <p>更新：∇θ J ≈ ∂Q/∂a · ∂μ/∂θ。</p>
            </CardContent>
          </Card>
          <div className="flex gap-2">
            <Button onClick={runEpisode} className="flex-1">训练 1 回合</Button>
            <Button onClick={reset} variant="outline" className="flex-1">重置</Button>
          </div>
        </div>
      </div>
    </InteractiveDemo>
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
