import { useMemo, useState } from 'react';
import { Activity, AlertTriangle, RefreshCw, ShieldAlert } from 'lucide-react';
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
import AlgorithmPlayer from '@/components/AlgorithmPlayer';
import LineChart from '@/components/LineChart';
import ConceptAccordion from '@/components/ConceptAccordion';
import {
  EPISODIC_PATH_CONFIG,
  ACTION_NAMES,
  type GridWorldConfig,
  type Policy,
  type Action,
  solveStateValues,
  computeQValues,
  actionValueToStateValue,
} from '@/lib/rl/gridworld';
import {
  qac,
  a2c,
  offPolicyActorCritic,
  qBasedOffPolicyActorCritic,
  deterministicActorCriticEpisode,
  policyWeightedStateValues,
  effectiveSampleSize,
  klDivergence,
  movingAverage,
  type ACUpdateRecord,
  type ACResult,
  type ACMetricSeries,
  type DeterministicACStep,
} from '@/lib/rl/actorCritic';

const H_OPTIONS = [10, 20, 30, 50, 100, 200];

type TabKey = 'overview' | 'qac' | 'baseline' | 'a2c' | 'offpolicy' | 'deterministic';
type TaskType = 'episodic' | 'continuing';

function buildConfig(base: GridWorldConfig, taskType: TaskType): GridWorldConfig {
  return { ...base, taskType };
}

function policyHistoryFromResult(result: ACResult): Policy[] {
  if (result.updates.length === 0) return result.finalPolicy ? [result.finalPolicy] : [];
  const firstFull = result.updates[0].actorFullPolicyBefore;
  const histories: number[][][] = firstFull ? [firstFull] : [result.updates[0].actorPolicyBefore.map(() => result.updates[0].actorPolicyBefore)];
  let lastEpisode = 0;
  result.updates.forEach((u) => {
    if (u.episode !== lastEpisode) {
      histories.push(u.actorFullPolicyBefore ?? histories[histories.length - 1]);
      lastEpisode = u.episode;
    }
    histories[histories.length - 1] = u.actorFullPolicyAfter ?? histories[histories.length - 1];
  });
  return histories;
}

function computeMetricSeries(result: ACResult): ACMetricSeries[] {
  const policies = policyHistoryFromResult(result);
  const perEpisode: Record<
    number,
    {
      returns: number[];
      lengths: number[];
      tdErrors: number[];
      actorNorms: number[];
      criticUpdates: number[];
      entropies: number[];
      kls: number[];
    }
  > = {};

  result.updates.forEach((u) => {
    const e = u.episode;
    if (!perEpisode[e]) {
      perEpisode[e] = { returns: [], lengths: [], tdErrors: [], actorNorms: [], criticUpdates: [], entropies: [], kls: [] };
    }
    perEpisode[e].tdErrors.push(u.tdError);
    perEpisode[e].actorNorms.push(Math.sqrt(u.actorDelta.reduce((s, x) => s + x * x, 0)));
    const critUpdate = u.vAfter
      ? Math.abs(u.criticEstimateAfter - u.criticEstimateBefore)
      : Math.abs(u.criticEstimateAfter - u.criticEstimateBefore);
    perEpisode[e].criticUpdates.push(critUpdate);
  });

  result.episodes.forEach((ep, i) => {
    const e = i + 1;
    if (!perEpisode[e]) {
      perEpisode[e] = { returns: [], lengths: [], tdErrors: [], actorNorms: [], criticUpdates: [], entropies: [], kls: [] };
    }
    perEpisode[e].returns.push(ep.cumulativeReward);
    perEpisode[e].lengths.push(ep.episodeLength);
  });

  policies.forEach((pol, i) => {
    const e = i;
    if (!perEpisode[e]) {
      perEpisode[e] = { returns: [], lengths: [], tdErrors: [], actorNorms: [], criticUpdates: [], entropies: [], kls: [] };
    }
    const ent = pol.reduce((sum, dist) => {
      return sum - dist.reduce((s, p) => (p > 0 ? s + p * Math.log(p) : s), 0);
    }, 0) / pol.length;
    perEpisode[e].entropies.push(ent);
    if (i > 0) {
      const kl = pol.reduce((sum, dist, s) => sum + klDivergence(dist, policies[i - 1][s]), 0);
      perEpisode[e].kls.push(kl);
    }
  });

  const episodes = Object.keys(perEpisode)
    .map(Number)
    .sort((a, b) => a - b);
  const returns = episodes.map((e) => perEpisode[e].returns[0] ?? 0);
  const ma = movingAverage(returns, 10);

  const avg = (arr: number[]) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);

  return episodes.map((e, i) => ({
    episode: e,
    returnValue: returns[i],
    movingAverage: ma[i],
    length: perEpisode[e].lengths[0] ?? 0,
    success: (perEpisode[e].lengths[0] ?? 0) > 0 && perEpisode[e].returns[0] > 0 ? 1 : 0,
    tdError: avg(perEpisode[e].tdErrors),
    actorGradientNorm: avg(perEpisode[e].actorNorms),
    criticUpdateNorm: avg(perEpisode[e].criticUpdates),
    entropy: avg(perEpisode[e].entropies),
    kl: avg(perEpisode[e].kls),
  }));
}

export default function Chapter10AcPage() {
  const [tab, setTab] = useState<TabKey>('overview');

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Activity className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">第 10 章 Actor-Critic 方法</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Actor 产生动作，Critic 评价动作，二者协同学习策略与价值函数。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <Tabs value={tab} onValueChange={(v) => setTab(v as TabKey)}>
        <TabsList className="grid w-full grid-cols-3 md:grid-cols-6">
          <TabsTrigger value="overview">协作概览</TabsTrigger>
          <TabsTrigger value="qac">QAC</TabsTrigger>
          <TabsTrigger value="baseline">Advantage</TabsTrigger>
          <TabsTrigger value="a2c">A2C</TabsTrigger>
          <TabsTrigger value="offpolicy">异策略 AC</TabsTrigger>
          <TabsTrigger value="deterministic">确定性 PG</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-4">
          <OverviewDemo />
        </TabsContent>
        <TabsContent value="qac" className="mt-4">
          <QacDemo />
        </TabsContent>
        <TabsContent value="baseline" className="mt-4">
          <BaselineAdvantageDemo />
        </TabsContent>
        <TabsContent value="a2c" className="mt-4">
          <A2cDemo />
        </TabsContent>
        <TabsContent value="offpolicy" className="mt-4">
          <OffPolicyDemo />
        </TabsContent>
        <TabsContent value="deterministic" className="mt-4">
          <DeterministicAcDemo />
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
                  <li>A2C 用 TD error 作为 advantage 的单步样本估计。</li>
                  <li>异策略 AC 用重要性采样比 ρ 修正动作分布，但不自动修正状态分布。</li>
                  <li>确定性策略梯度适用于连续动作，需要额外探索机制。</li>
                </ul>
              ),
            },
            {
              id: 'qa1',
              title: 'Q: A2C 的 TD error 为什么能近似 advantage？',
              content:
                '当 Critic 等于真实 V_π 时，E[δ | s,a] = Q_π(s,a) − V_π(s) = A_π(s,a)。Critic 不准确时，TD error 只是带噪声的 advantage 近似。',
            },
            {
              id: 'qa2',
              title: 'Q: 异策略 AC 用 ρ 修正了什么？',
              content:
                'ρ_t = π(A_t|S_t)/β(A_t|S_t) 修正了单步动作采样分布，使目标策略的梯度期望成立。但它不自动修正状态访问分布的偏差；是否完全修正取决于具体算法目标。',
            },
          ]}
        />
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared UI helpers
// ---------------------------------------------------------------------------

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

function MetricPanel({ series, diverged, divergenceStep }: { series: ACMetricSeries[]; diverged: boolean; divergenceStep?: number }) {
  return (
    <div className="space-y-4">
      {diverged && (
        <div className="flex items-center gap-2 text-red-600 bg-red-50 p-3 rounded-lg border border-red-200 text-sm">
          <AlertTriangle className="w-4 h-4" />
          检测到数值不稳定（发散），大约在第 {divergenceStep} 步。请减小学习率或增大 Critic 学习率。
        </div>
      )}
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500">平均回报</div>
          <div className="font-mono font-semibold">
            {series.length > 0 ? series[series.length - 1].returnValue.toFixed(2) : '—'}
          </div>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500">移动平均回报</div>
          <div className="font-mono font-semibold">
            {series.length > 0 ? series[series.length - 1].movingAverage.toFixed(2) : '—'}
          </div>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500">平均 TD error</div>
          <div className="font-mono font-semibold">
            {series.length > 0 ? series[series.length - 1].tdError.toFixed(3) : '—'}
          </div>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500">策略熵</div>
          <div className="font-mono font-semibold">
            {series.length > 0 ? series[series.length - 1].entropy.toFixed(3) : '—'}
          </div>
        </div>
      </div>

      <LineChart
        data={series as any}
        xKey="episode"
        xLabel="回合"
        yLabel="回报"
        series={[
          { key: 'returnValue', name: 'Episode Return', color: '#2563eb' },
          { key: 'movingAverage', name: 'Moving Average', color: '#ef4444' },
        ]}
        height={160}
      />
      <LineChart
        data={series as any}
        xKey="episode"
        xLabel="回合"
        yLabel="指标"
        series={[
          { key: 'actorGradientNorm', name: 'Actor Gradient Norm', color: '#2563eb' },
          { key: 'criticUpdateNorm', name: 'Critic Update Norm', color: '#22c55e' },
        ]}
        height={140}
      />
    </div>
  );
}

function TransitionCard({ record }: { record: ACUpdateRecord }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
      <div className="bg-gray-50 rounded p-2">
        <div className="text-gray-500 text-xs">S_t</div>
        <div className="font-mono">s{record.state + 1}</div>
      </div>
      <div className="bg-gray-50 rounded p-2">
        <div className="text-gray-500 text-xs">A_t</div>
        <div className="font-mono">{ACTION_NAMES[record.action]}</div>
      </div>
      <div className="bg-gray-50 rounded p-2">
        <div className="text-gray-500 text-xs">R_{'{t+1}'}</div>
        <div className="font-mono">{record.reward.toFixed(2)}</div>
      </div>
      <div className="bg-gray-50 rounded p-2">
        <div className="text-gray-500 text-xs">S_{'{t+1}'}</div>
        <div className="font-mono">s{record.nextState + 1}</div>
      </div>
    </div>
  );
}

function CriticCard({ record }: { record: ACUpdateRecord }) {
  return (
    <div className="space-y-2 text-sm">
      <div className="grid grid-cols-3 gap-2">
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500 text-xs">old estimate</div>
          <div className="font-mono">{record.criticEstimateBefore.toFixed(4)}</div>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500 text-xs">bootstrap</div>
          <div className="font-mono">{record.criticBootstrap.toFixed(4)}</div>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500 text-xs">target</div>
          <div className="font-mono">{record.criticTarget.toFixed(4)}</div>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-blue-50 rounded p-2 border border-blue-100">
          <div className="text-blue-700 text-xs">TD error</div>
          <div className="font-mono font-semibold">{record.tdError.toFixed(4)}</div>
        </div>
        <div className="bg-green-50 rounded p-2 border border-green-100">
          <div className="text-green-700 text-xs">new estimate</div>
          <div className="font-mono font-semibold">{record.criticEstimateAfter.toFixed(4)}</div>
        </div>
      </div>
    </div>
  );
}

function ActorCard({ record }: { record: ACUpdateRecord }) {
  return (
    <div className="space-y-2 text-sm">
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500 text-xs">actor weight</div>
          <div className="font-mono">{record.actorWeight.toFixed(4)}</div>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500 text-xs">score gradient norm</div>
          <div className="font-mono">
            {Math.sqrt(record.scoreGradient.reduce((s, x) => s + x * x, 0)).toFixed(4)}
          </div>
        </div>
      </div>
      <div className="bg-gray-50 rounded p-2">
        <div className="text-gray-500 text-xs">policy before / after</div>
        <div className="grid grid-cols-5 gap-1 text-center">
          {record.actorPolicyBefore.map((p, a) => (
            <div key={a} className="space-y-1">
              <div className="text-[10px] text-gray-500">{ACTION_NAMES[a]}</div>
              <div className="font-mono text-xs">{p.toFixed(2)}</div>
              <div
                className={`font-mono text-xs ${
                  record.actorPolicyAfter[a] > p ? 'text-green-600' : record.actorPolicyAfter[a] < p ? 'text-red-600' : 'text-gray-600'
                }`}
              >
                {record.actorPolicyAfter[a].toFixed(2)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DiscreteControlPanel({
  seed,
  setSeed,
  horizonH,
  setHorizonH,
  taskType,
  setTaskType,
  actorAlpha,
  setActorAlpha,
  criticAlpha,
  setCriticAlpha,
  episodes,
  setEpisodes,
  children,
}: {
  seed: number;
  setSeed: (v: number) => void;
  horizonH: number;
  setHorizonH: (v: number) => void;
  taskType: TaskType;
  setTaskType: (v: TaskType) => void;
  actorAlpha: number;
  setActorAlpha: (v: number) => void;
  criticAlpha: number;
  setCriticAlpha: (v: number) => void;
  episodes: number;
  setEpisodes: (v: number) => void;
  children?: React.ReactNode;
}) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-sm text-gray-700 block mb-1">任务类型</label>
          <Select value={taskType} onValueChange={(v) => setTaskType(v as TaskType)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="episodic">episodic</SelectItem>
              <SelectItem value="continuing">truncated continuing</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <label className="text-sm text-gray-700 block mb-1">horizon H</label>
          <Select value={String(horizonH)} onValueChange={(v) => setHorizonH(Number(v))}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {H_OPTIONS.map((h) => (
                <SelectItem key={h} value={String(h)}>
                  {h}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex-1">
          <label className="text-sm text-gray-700 block mb-1">seed</label>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value) || 0)}
            className="w-full px-2 py-1 border rounded text-sm"
          />
        </div>
        <Button variant="outline" size="sm" onClick={() => setSeed(Math.floor(Math.random() * 10000))}>
          <RefreshCw className="w-4 h-4 mr-1" /> 换种子
        </Button>
      </div>
      <Param label="Actor α" value={actorAlpha} set={setActorAlpha} min={0.001} max={0.3} step={0.001} fixed={3} />
      <Param label="Critic α" value={criticAlpha} set={setCriticAlpha} min={0.001} max={0.5} step={0.001} fixed={3} />
      <Param label="训练回合数" value={episodes} set={setEpisodes} min={10} max={200} step={10} />
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 1. Overview: how Actor and Critic collaborate
// ---------------------------------------------------------------------------

function OverviewDemo() {
  const config = useMemo(() => buildConfig(EPISODIC_PATH_CONFIG, 'episodic'), []);
  const result = useMemo(
    () =>
      qac(config, {
        seed: 1,
        horizonH: 20,
        actorAlpha: 0.05,
        criticAlpha: 0.1,
        episodes: 1,
      }),
    [config]
  );
  const [step, setStep] = useState(0);
  const record = result.updates[Math.min(step, result.updates.length - 1)];

  return (
    <InteractiveDemo title="Actor 与 Critic 如何协作">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="space-y-4">
          <FormulaCard
            title="一次 transition 的协作流程"
            formula={
              <KaTeX
                math={String.raw`\begin{aligned}
S_t &\xrightarrow{\text{Actor } \pi_\theta} A_t \\
A_t &\xrightarrow{\text{环境}} R_{t+1}, S_{t+1} \\
S_{t+1} &\xrightarrow{\text{Critic } v_w} \text{bootstrap} \\
\delta_t &= R_{t+1} + \gamma v_w(S_{t+1}) - v_w(S_t) \\
\theta &\leftarrow \theta + \alpha_\theta \, \delta_t \, \nabla_\theta \log \pi_\theta(A_t|S_t)
\end{aligned}`}
                display
              />
            }
            description="Critic 提供评价信号（TD error），Actor 用它更新策略。"
          />
          {record && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">第 {step + 1} 步示例</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <TransitionCard record={record} />
                <div className="grid grid-cols-2 gap-3">
                  <CriticCard record={record} />
                  <ActorCard record={record} />
                </div>
              </CardContent>
            </Card>
          )}
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">概念地图</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <p>
                <strong>Actor</strong> 更新对象：策略参数 θ，目标是提高期望回报。
              </p>
              <p>
                <strong>Critic</strong> 更新对象：价值参数 w（V 或 Q），目标是更准确预测回报。
              </p>
              <p>
                <strong>两个学习率</strong>：α_θ 与 α_w，通常 Critic 需要更快跟踪当前策略。
              </p>
              <p>
                <strong>on-policy / off-policy</strong>：采样策略与目标策略是否相同决定是否需要重要性采样。
              </p>
              <p>
                <strong>stochastic / deterministic</strong>：策略输出动作分布还是确定性动作。
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">回放控制</CardTitle>
            </CardHeader>
            <CardContent>
              <AlgorithmPlayer maxStep={result.updates.length - 1} currentStep={step} onStepChange={setStep} />
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ---------------------------------------------------------------------------
// 2. QAC
// ---------------------------------------------------------------------------

function QacDemo() {
  const [seed, setSeed] = useState(42);
  const [horizonH, setHorizonH] = useState(30);
  const [taskType, setTaskType] = useState<TaskType>('episodic');
  const [actorAlpha, setActorAlpha] = useState(0.05);
  const [criticAlpha, setCriticAlpha] = useState(0.1);
  const [episodes, setEpisodes] = useState(80);
  const [valueMode, setValueMode] = useState<'actor' | 'greedy'>('actor');

  const config = useMemo(() => buildConfig(EPISODIC_PATH_CONFIG, taskType), [taskType]);
  const result = useMemo(
    () =>
      qac(config, {
        seed,
        horizonH,
        actorAlpha,
        criticAlpha,
        episodes,
      }),
    [config, seed, horizonH, actorAlpha, criticAlpha, episodes]
  );
  const [step, setStep] = useState(0);
  const record = result.updates[Math.min(step, result.updates.length - 1)];
  const metricSeries = useMemo(() => computeMetricSeries(result), [result]);

  const currentQ = record?.qAfter;
  const currentPolicy = record?.actorFullPolicyAfter;
  const values = useMemo(() => {
    if (!currentQ || !currentPolicy) return new Array(config.rows * config.cols).fill(0);
    return valueMode === 'actor'
      ? policyWeightedStateValues(currentQ, currentPolicy)
      : actionValueToStateValue(currentQ);
  }, [currentQ, currentPolicy, valueMode, config.rows, config.cols]);

  return (
    <InteractiveDemo title="QAC：Critic 估计 q(s,a) 驱动 Actor">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="space-y-4">
          <FormulaCard
            title="QAC 更新"
            formula={
              <KaTeX
                math={String.raw`\begin{aligned}
\delta_t &= R_{t+1} + \gamma q(S_{t+1}, A_{t+1}) - q(S_t, A_t) \\
q(S_t,A_t) &\leftarrow q(S_t,A_t) + \alpha_w \delta_t \\
\theta &\leftarrow \theta + \alpha_\theta \, q_{\text{before}}(S_t,A_t) \, \nabla_\theta \log \pi_\theta(A_t|S_t)
\end{aligned}`}
                display
              />
            }
            description="Actor 使用更新前的 q_before 作为权重，而不是更新后的 q_after。"
          />
          {record && (
            <>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">
                    第 {record.episode} 回合 · 第 {record.time + 1} 步
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <TransitionCard record={record} />
                  <div className="grid grid-cols-2 gap-3">
                    <CriticCard record={record} />
                    <ActorCard record={record} />
                  </div>
                </CardContent>
              </Card>
              <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600">值函数显示</span>
                  <Select value={valueMode} onValueChange={(v) => setValueMode(v as 'actor' | 'greedy')}>
                    <SelectTrigger className="w-40">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="actor">当前 Actor 策略价值 V_π</SelectItem>
                      <SelectItem value="greedy">greedy-derived value</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <GridWorld config={config} policy={currentPolicy} values={values} showValues className="max-w-full" />
                <p className="mt-2 text-xs text-gray-600">
                  {valueMode === 'actor'
                    ? 'V_π(s) = Σ_a π(a|s) q(s,a)，对应当前 Actor 的随机策略。'
                    : 'max_a q(s,a)，对应贪婪策略的值。'}
                </p>
              </div>
            </>
          )}
          <MetricPanel series={metricSeries} diverged={result.diverged} divergenceStep={result.divergenceStep} />
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">超参数</CardTitle>
            </CardHeader>
            <CardContent>
              <DiscreteControlPanel
                seed={seed}
                setSeed={setSeed}
                horizonH={horizonH}
                setHorizonH={setHorizonH}
                taskType={taskType}
                setTaskType={setTaskType}
                actorAlpha={actorAlpha}
                setActorAlpha={setActorAlpha}
                criticAlpha={criticAlpha}
                setCriticAlpha={setCriticAlpha}
                episodes={episodes}
                setEpisodes={setEpisodes}
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">回放控制</CardTitle>
            </CardHeader>
            <CardContent>
              <AlgorithmPlayer maxStep={result.updates.length - 1} currentStep={step} onStepChange={setStep} />
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ---------------------------------------------------------------------------
// 3. Baseline invariance & Advantage
// ---------------------------------------------------------------------------

function BaselineAdvantageDemo() {
  const config = useMemo(() => buildConfig(EPISODIC_PATH_CONFIG, 'episodic'), []);
  const result = useMemo(
    () =>
      a2c(config, {
        seed: 1,
        horizonH: 30,
        actorAlpha: 0.05,
        criticAlpha: 0.1,
        episodes: 50,
      }),
    [config]
  );
  const [step, setStep] = useState(0);
  const record = result.updates[Math.min(step, result.updates.length - 1)];
  const metricSeries = useMemo(() => computeMetricSeries(result), [result]);

  return (
    <InteractiveDemo title="Baseline 不变性与 Advantage 估计">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="space-y-4">
          <FormulaCard
            title="TD error 作为 advantage 的单步估计"
            formula={
              <KaTeX
                math={String.raw`\delta_t = R_{t+1} + \gamma v(S_{t+1}) - v(S_t) \approx A_\pi(S_t, A_t)`}
                display
              />
            }
            description="当 Critic 等于真实 V_π 时，E[δ | s,a] = Q_π(s,a) − V_π(s) = A_π(s,a)。Critic 不准确时，它只是近似。"
          />
          {record && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">
                  第 {record.episode} 回合 · 第 {record.time + 1} 步
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <TransitionCard record={record} />
                <div className="grid grid-cols-2 gap-3">
                  <CriticCard record={record} />
                  <ActorCard record={record} />
                </div>
              </CardContent>
            </Card>
          )}
          <ExactAdvantagePanel config={config} />
          <MetricPanel series={metricSeries} diverged={result.diverged} divergenceStep={result.divergenceStep} />
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">说明</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <p>
                任何不依赖当前动作的 baseline b(s) 都满足
                <KaTeX math={String.raw`\sum_a \pi(a|s) \nabla\log\pi(a|s) b(s) = 0`} display={false} />
                ，因此不改变期望策略梯度。
              </p>
              <p>
                用 TD error 替代回报作为 Actor 权重，相当于把 baseline 设为了状态值函数，从而降低方差。
              </p>
              <p>
                但 TD error 只有在 Critic 准确时才是 advantage 的无偏估计；训练初期它带有明显的估计噪声。
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">回放控制</CardTitle>
            </CardHeader>
            <CardContent>
              <AlgorithmPlayer maxStep={result.updates.length - 1} currentStep={step} onStepChange={setStep} />
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

function ExactAdvantagePanel({ config }: { config: GridWorldConfig }) {
  const [state, setState] = useState(0);
  const [action, setAction] = useState<Action>(1);
  const numStates = config.rows * config.cols;

  const uniformPolicy = useMemo(
    () => Array.from({ length: numStates }, () => Array.from({ length: 5 }, () => 1 / 5)),
    [numStates]
  );
  const trueV = useMemo(() => solveStateValues(uniformPolicy, config), [uniformPolicy, config]);
  const trueQ = useMemo(() => computeQValues(trueV, config), [trueV, config]);
  const trueAdvantage = trueQ[state][action] - trueV[state];

  const estimated = useMemo(() => {
    const samples: number[] = [];
    for (let seed = 1; seed <= 300; seed++) {
      const res = a2c(config, { seed, horizonH: 1, actorAlpha: 0, criticAlpha: 0, episodes: 1 });
      const u = res.updates.find((up) => up.state === state && up.action === action);
      if (u) {
        const bootstrap = u.done ? 0 : trueV[u.nextState];
        samples.push(u.reward + config.gamma * bootstrap - trueV[state]);
      }
    }
    return samples;
  }, [config, state, action, trueV]);

  const mean = estimated.length ? estimated.reduce((a, b) => a + b, 0) / estimated.length : 0;
  const std = estimated.length
    ? Math.sqrt(estimated.reduce((s, v) => s + (v - mean) ** 2, 0) / estimated.length)
    : 0;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">精确对照：TD error 的期望 vs 真实 Advantage</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-sm text-gray-700 block mb-1">状态</label>
            <Select value={String(state)} onValueChange={(v) => setState(Number(v))}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Array.from({ length: numStates }, (_, s) => (
                  <SelectItem key={s} value={String(s)}>
                    s{s + 1}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="text-sm text-gray-700 block mb-1">动作</label>
            <Select value={String(action)} onValueChange={(v) => setAction(Number(v) as Action)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {ACTION_NAMES.map((name, a) => (
                  <SelectItem key={a} value={String(a)}>
                    {name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-gray-50 rounded p-2">
            <div className="text-gray-500 text-xs">真实 A_π(s,a)</div>
            <div className="font-mono">{trueAdvantage.toFixed(4)}</div>
          </div>
          <div className="bg-gray-50 rounded p-2">
            <div className="text-gray-500 text-xs">样本 δ 均值</div>
            <div className="font-mono">{mean.toFixed(4)}</div>
          </div>
          <div className="bg-gray-50 rounded p-2">
            <div className="text-gray-500 text-xs">样本 δ 标准差</div>
            <div className="font-mono">{std.toFixed(4)}</div>
          </div>
        </div>
        <p className="text-xs text-gray-600">
          用真实 V_π 作为 Critic，对 (s,a) 重复采样得到 TD error。其样本均值应接近真实 advantage，但有限样本存在随机波动。
        </p>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4. A2C
// ---------------------------------------------------------------------------

function A2cDemo() {
  const [seed, setSeed] = useState(42);
  const [horizonH, setHorizonH] = useState(30);
  const [taskType, setTaskType] = useState<TaskType>('episodic');
  const [actorAlpha, setActorAlpha] = useState(0.05);
  const [criticAlpha, setCriticAlpha] = useState(0.1);
  const [episodes, setEpisodes] = useState(80);

  const config = useMemo(() => buildConfig(EPISODIC_PATH_CONFIG, taskType), [taskType]);
  const result = useMemo(
    () =>
      a2c(config, {
        seed,
        horizonH,
        actorAlpha,
        criticAlpha,
        episodes,
      }),
    [config, seed, horizonH, actorAlpha, criticAlpha, episodes]
  );
  const [step, setStep] = useState(0);
  const record = result.updates[Math.min(step, result.updates.length - 1)];
  const metricSeries = useMemo(() => computeMetricSeries(result), [result]);

  const values = record?.vAfter ?? new Array(config.rows * config.cols).fill(0);
  const policy = record?.actorFullPolicyAfter ?? Array.from({ length: config.rows * config.cols }, () => Array.from({ length: 5 }, () => 1 / 5));

  return (
    <InteractiveDemo title="A2C：TD error 作为 Advantage">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="space-y-4">
          <FormulaCard
            title="A2C 更新"
            formula={
              <KaTeX
                math={String.raw`\begin{aligned}
\delta_t &= R_{t+1} + \gamma v(S_{t+1}) - v(S_t) \\
v(S_t) &\leftarrow v(S_t) + \alpha_w \delta_t \\
\theta &\leftarrow \theta + \alpha_\theta \, \delta_t \, \nabla_\theta \log \pi_\theta(A_t|S_t)
\end{aligned}`}
                display
              />
            }
            description="TD error 是 advantage 的单步样本估计。Critic 越准确，它越接近真实 A_π。"
          />
          {record && (
            <>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">
                    第 {record.episode} 回合 · 第 {record.time + 1} 步
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <TransitionCard record={record} />
                  <div className="grid grid-cols-2 gap-3">
                    <CriticCard record={record} />
                    <ActorCard record={record} />
                  </div>
                </CardContent>
              </Card>
              <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
                <GridWorld config={config} policy={policy} values={values} showValues className="max-w-full" />
                <p className="mt-2 text-xs text-gray-600">蓝色高亮为当前状态，红色高亮为当前动作。</p>
              </div>
            </>
          )}
          <MetricPanel series={metricSeries} diverged={result.diverged} divergenceStep={result.divergenceStep} />
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">超参数</CardTitle>
            </CardHeader>
            <CardContent>
              <DiscreteControlPanel
                seed={seed}
                setSeed={setSeed}
                horizonH={horizonH}
                setHorizonH={setHorizonH}
                taskType={taskType}
                setTaskType={setTaskType}
                actorAlpha={actorAlpha}
                setActorAlpha={setActorAlpha}
                criticAlpha={criticAlpha}
                setCriticAlpha={setCriticAlpha}
                episodes={episodes}
                setEpisodes={setEpisodes}
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">回放控制</CardTitle>
            </CardHeader>
            <CardContent>
              <AlgorithmPlayer maxStep={result.updates.length - 1} currentStep={step} onStepChange={setStep} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">学习率比例实验</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <p>当前 ratio α_actor / α_critic = {(actorAlpha / criticAlpha).toFixed(2)}。</p>
              <p>
                Critic 通常需要比 Actor 更快地跟踪当前策略，但具体时间尺度条件依赖算法与理论设定。若训练发散，通常应减小 Actor α 或增大 Critic α。
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ---------------------------------------------------------------------------
// 5. Off-policy AC
// ---------------------------------------------------------------------------

function OffPolicyDemo() {
  const [seed, setSeed] = useState(42);
  const [horizonH, setHorizonH] = useState(30);
  const [taskType, setTaskType] = useState<TaskType>('episodic');
  const [actorAlpha, setActorAlpha] = useState(0.05);
  const [criticAlpha, setCriticAlpha] = useState(0.1);
  const [epsilon, setEpsilon] = useState(0.5);
  const [episodes, setEpisodes] = useState(80);
  const [variant, setVariant] = useState<'textbook' | 'extended'>('textbook');

  const config = useMemo(() => buildConfig(EPISODIC_PATH_CONFIG, taskType), [taskType]);
  const result = useMemo(() => {
    const options = { seed, horizonH, actorAlpha, criticAlpha, episodes, epsilon };
    try {
      return variant === 'textbook'
        ? offPolicyActorCritic(config, options)
        : qBasedOffPolicyActorCritic(config, options);
    } catch (err) {
      return null;
    }
  }, [config, seed, horizonH, actorAlpha, criticAlpha, episodes, epsilon, variant]);

  const [step, setStep] = useState(0);
  const record = result ? result.updates[Math.min(step, result.updates.length - 1)] : undefined;
  const metricSeries = useMemo(() => (result ? computeMetricSeries(result) : []), [result]);

  const values = useMemo(() => {
    if (!record) return new Array(config.rows * config.cols).fill(0);
    if (variant === 'textbook' && record.vAfter) return record.vAfter;
    if (variant === 'extended' && record.qAfter) return actionValueToStateValue(record.qAfter);
    return new Array(config.rows * config.cols).fill(0);
  }, [record, variant, config.rows, config.cols]);

  const policy = record?.actorFullPolicyAfter ?? Array.from({ length: config.rows * config.cols }, () => Array.from({ length: 5 }, () => 1 / 5));

  const rhoStats = useMemo(() => {
    if (!result) return null;
    const rhos = result.updates.map((u) => u.rho ?? 1);
    const maxRho = Math.max(...rhos);
    const meanRho = rhos.reduce((a, b) => a + b, 0) / rhos.length;
    const clipped = rhos.filter((r) => r > 10).length / rhos.length;
    const ess = effectiveSampleSize(rhos);
    return { maxRho, meanRho, clipped, ess };
  }, [result]);

  return (
    <InteractiveDemo title="异策略 Actor-Critic：重要性采样修正">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="space-y-4">
          <FormulaCard
            title="重要性采样比"
            formula={
              <KaTeX
                math={String.raw`\rho_t = \frac{\pi_\theta(A_t|S_t)}{\beta(A_t|S_t)}, \quad \theta \leftarrow \theta + \alpha_\theta \, \rho_t \, \delta_t \, \nabla_\theta \log \pi_\theta(A_t|S_t)`}
                display
              />
            }
            description="ρ 修正单步动作采样分布，使目标策略梯度期望成立；但它不自动修正状态访问分布。"
          />
          {!result && (
            <div className="flex items-center gap-2 text-red-600 bg-red-50 p-3 rounded-lg border border-red-200 text-sm">
              <AlertTriangle className="w-4 h-4" />
              行为策略未覆盖目标策略动作。请增大 ε 或检查策略支持。
            </div>
          )}
          {record && (
            <>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">
                    第 {record.episode} 回合 · 第 {record.time + 1} 步
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <TransitionCard record={record} />
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="bg-gray-50 rounded p-2">
                      <div className="text-gray-500 text-xs">target π(A|S)</div>
                      <div className="font-mono">{record.targetProb?.toFixed(4)}</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2">
                      <div className="text-gray-500 text-xs">behavior β(A|S)</div>
                      <div className="font-mono">{record.behaviorProb?.toFixed(4)}</div>
                    </div>
                    <div className="bg-blue-50 rounded p-2 border border-blue-100">
                      <div className="text-blue-700 text-xs">ρ = π/β</div>
                      <div className="font-mono font-semibold">{record.rho?.toFixed(4)}</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2">
                      <div className="text-gray-500 text-xs">校正前 actor weight</div>
                      <div className="font-mono">{(record.actorWeight / (record.rho ?? 1)).toFixed(4)}</div>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <CriticCard record={record} />
                    <ActorCard record={record} />
                  </div>
                </CardContent>
              </Card>
              <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
                <GridWorld config={config} policy={policy} values={values} showValues className="max-w-full" />
              </div>
              {rhoStats && (
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">ρ 统计</CardTitle>
                  </CardHeader>
                  <CardContent className="grid grid-cols-4 gap-3 text-sm">
                    <div className="bg-gray-50 rounded p-2">
                      <div className="text-gray-500 text-xs">max ρ</div>
                      <div className="font-mono">{rhoStats.maxRho.toFixed(2)}</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2">
                      <div className="text-gray-500 text-xs">mean ρ</div>
                      <div className="font-mono">{rhoStats.meanRho.toFixed(2)}</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2">
                      <div className="text-gray-500 text-xs">clipped (ρ&gt;10)</div>
                      <div className="font-mono">{(rhoStats.clipped * 100).toFixed(1)}%</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2">
                      <div className="text-gray-500 text-xs">ESS</div>
                      <div className="font-mono">{rhoStats.ess.toFixed(2)}</div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}
          {result && <MetricPanel series={metricSeries} diverged={result.diverged} divergenceStep={result.divergenceStep} />}
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">超参数</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm text-gray-700 block mb-1">算法版本</label>
                <Select value={variant} onValueChange={(v) => setVariant(v as 'textbook' | 'extended')}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="textbook">教材版：V-based + ρ</SelectItem>
                    <SelectItem value="extended">扩展版：Q-based + ρ</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <DiscreteControlPanel
                seed={seed}
                setSeed={setSeed}
                horizonH={horizonH}
                setHorizonH={setHorizonH}
                taskType={taskType}
                setTaskType={setTaskType}
                actorAlpha={actorAlpha}
                setActorAlpha={setActorAlpha}
                criticAlpha={criticAlpha}
                setCriticAlpha={setCriticAlpha}
                episodes={episodes}
                setEpisodes={setEpisodes}
              >
                <Param label="行为策略 ε" value={epsilon} set={setEpsilon} min={0.05} max={1} step={0.05} fixed={2} />
              </DiscreteControlPanel>
            </CardContent>
          </Card>
          {result && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">回放控制</CardTitle>
              </CardHeader>
              <CardContent>
                <AlgorithmPlayer maxStep={result.updates.length - 1} currentStep={step} onStepChange={setStep} />
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ---------------------------------------------------------------------------
// 6. Deterministic policy gradient toy
// ---------------------------------------------------------------------------

function DeterministicAcDemo() {
  const [seed, setSeed] = useState(1);
  const [horizonH, setHorizonH] = useState(20);
  const [actorAlpha, setActorAlpha] = useState(0.05);
  const [criticAlpha, setCriticAlpha] = useState(0.1);
  const [gamma, setGamma] = useState(0.9);
  const [mode, setMode] = useState<'pure' | 'explore'>('pure');
  const [theta0, setTheta0] = useState(0.5);
  const [theta1, setTheta1] = useState(0.0);
  const [w, setW] = useState<number[]>([0, 0, 0, 0, 0, 0]);
  const [episodeCount, setEpisodeCount] = useState(0);
  const [history, setHistory] = useState<DeterministicACStep[]>([]);
  const [step, setStep] = useState(0);

  const env = {
    step: (state: number, action: number) => {
      const nextState = Math.max(-1, Math.min(1, state + 0.3 * action));
      return { nextState, reward: -(nextState * nextState), done: Math.abs(nextState) < 0.05 };
    },
    terminal: (state: number) => Math.abs(state) < 0.05,
  };

  function features(s: number, a: number): number[] {
    return [1, s, a, s * s, s * a, a * a];
  }

  function qValue(s: number, a: number, weights: number[]): number {
    return features(s, a).reduce((sum, f, i) => sum + f * weights[i], 0);
  }

  function mu(s: number, theta: number[]): number {
    return Math.tanh(theta[0] * s + theta[1]);
  }

  function gradMu(s: number, theta: number[]): number[] {
    const m = mu(s, theta);
    const factor = 1 - m * m;
    return [factor * s, factor];
  }

  function gradQWrtA(s: number, a: number, weights: number[]): number {
    return weights[2] + weights[4] * s + 2 * weights[5] * a;
  }

  function runEpisode() {
    const res = deterministicActorCriticEpisode(env, -0.8, w, [theta0, theta1], {
      seed,
      horizonH,
      actorAlpha,
      criticAlpha,
      gamma,
      qValueWithWeights: qValue,
      muWithTheta: mu,
      gradMuWithTheta: gradMu,
      gradQWrtA,
      criticFeatures: features,
      explorationNoiseStd: mode === 'explore' ? 0.2 : 0,
    });
    setW(res.finalW);
    setTheta0(res.finalTheta[0]);
    setTheta1(res.finalTheta[1]);
    setHistory(res.steps);
    setStep(0);
    setEpisodeCount((c) => c + 1);
  }

  function reset() {
    setTheta0(0.5);
    setTheta1(0.0);
    setW([0, 0, 0, 0, 0, 0]);
    setEpisodeCount(0);
    setHistory([]);
    setStep(0);
  }

  const currentStep = history[Math.min(step, history.length - 1)];
  const stateForPlot = 0.3;
  const plotActions = Array.from({ length: 41 }, (_, i) => -1 + (2 * i) / 40);
  const plotValues = plotActions.map((a) => qValue(stateForPlot, a, w));
  const maxQ = Math.max(...plotValues);
  const minQ = Math.min(...plotValues);
  const range = maxQ - minQ || 1;
  const actionForPlot = mu(stateForPlot, [theta0, theta1]);

  return (
    <InteractiveDemo title="确定性策略梯度教学玩具">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="space-y-4">
          <FormulaCard
            title="确定性策略梯度"
            formula={
              <KaTeX
                math={String.raw`\nabla_\theta J \approx \nabla_a Q(s,a,w)\big|_{a=\mu_\theta(s)} \, \nabla_\theta \mu_\theta(s)`}
                display
              />
            }
            description="这不是 DDPG：没有 replay buffer、target network 或 soft update，仅用于理解确定性 Actor 如何沿 Critic 梯度移动。"
          />

          <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
            <div className="text-sm text-gray-600 mb-2">固定状态 s = {stateForPlot.toFixed(1)} 下的 Critic Q(s,a)</div>
            <svg width={360} height={220} className="bg-white rounded-lg border border-gray-200 w-full">
              {plotActions.map((a, i) => {
                const x = 30 + ((a + 1) / 2) * 300;
                const y = 190 - ((plotValues[i] - minQ) / range) * 160;
                return <circle key={i} cx={x} cy={y} r={2} fill="#2563eb" />;
              })}
              <circle
                cx={30 + ((actionForPlot + 1) / 2) * 300}
                cy={190 - ((qValue(stateForPlot, actionForPlot, w) - minQ) / range) * 160}
                r={6}
                fill="#ef4444"
              />
              <text x={180} y={210} textAnchor="middle" fontSize={10} fill="#6b7280">
                动作 a
              </text>
              <text x={10} y={15} fontSize={10} fill="#6b7280">
                Q
              </text>
            </svg>
            <p className="mt-2 text-sm text-gray-600">
              蓝点为 Q(s,a) 曲线，红点为当前确定性策略 μ(s)。Actor 沿 Q 增大方向调整 μ。
            </p>
          </div>

          {currentStep && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">第 {step + 1} 步</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="grid grid-cols-3 gap-2">
                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-gray-500 text-xs">s</div>
                    <div className="font-mono">{currentStep.state.toFixed(3)}</div>
                  </div>
                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-gray-500 text-xs">a = μ(s)</div>
                    <div className="font-mono">{currentStep.action.toFixed(3)}</div>
                  </div>
                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-gray-500 text-xs">r</div>
                    <div className="font-mono">{currentStep.reward.toFixed(4)}</div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-gray-500 text-xs">target</div>
                    <div className="font-mono">{currentStep.target.toFixed(4)}</div>
                  </div>
                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-gray-500 text-xs">prediction</div>
                    <div className="font-mono">{currentStep.prediction.toFixed(4)}</div>
                  </div>
                  <div className="bg-blue-50 rounded p-2 border border-blue-100">
                    <div className="text-blue-700 text-xs">TD error</div>
                    <div className="font-mono font-semibold">{currentStep.tdError.toFixed(4)}</div>
                  </div>
                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-gray-500 text-xs">critic loss</div>
                    <div className="font-mono">{(currentStep.tdError * currentStep.tdError).toFixed(4)}</div>
                  </div>
                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-gray-500 text-xs">∂Q/∂a</div>
                    <div className="font-mono">{currentStep.dqda.toFixed(4)}</div>
                  </div>
                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-gray-500 text-xs">distance to target</div>
                    <div className="font-mono">{Math.abs(currentStep.state).toFixed(4)}</div>
                  </div>
                </div>
                <div className="bg-gray-50 rounded p-2">
                  <div className="text-gray-500 text-xs">actor gradient</div>
                  <div className="font-mono">
                    [{currentStep.actorGradient.map((g) => g.toFixed(4)).join(', ')}]
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {history.length > 0 && (
            <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
              <LineChart
                data={history.map((h, i) => ({ step: i, tdError: h.tdError, dqda: h.dqda, distance: Math.abs(h.state) })) as any}
                xKey="step"
                xLabel="步数"
                yLabel="数值"
                series={[
                  { key: 'tdError', name: 'TD error', color: '#2563eb' },
                  { key: 'distance', name: '|s|', color: '#ef4444' },
                ]}
                height={160}
              />
            </div>
          )}
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">超参数</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm text-gray-700 block mb-1">模式</label>
                <Select value={mode} onValueChange={(v) => setMode(v as 'pure' | 'explore')}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="pure">纯 DPG 机制演示</SelectItem>
                    <SelectItem value="explore">加 Gaussian exploration noise</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex-1">
                  <label className="text-sm text-gray-700 block mb-1">seed</label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value) || 0)}
                    className="w-full px-2 py-1 border rounded text-sm"
                  />
                </div>
                <Button variant="outline" size="sm" onClick={() => setSeed(Math.floor(Math.random() * 10000))}>
                  <RefreshCw className="w-4 h-4 mr-1" /> 换种子
                </Button>
              </div>
              <div>
                <label className="text-sm text-gray-700 block mb-1">horizon H</label>
                <Select value={String(horizonH)} onValueChange={(v) => setHorizonH(Number(v))}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {H_OPTIONS.map((h) => (
                      <SelectItem key={h} value={String(h)}>
                        {h}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <Param label="Actor α" value={actorAlpha} set={setActorAlpha} min={0.001} max={0.3} step={0.001} fixed={3} />
              <Param label="Critic α" value={criticAlpha} set={setCriticAlpha} min={0.001} max={0.5} step={0.001} fixed={3} />
              <Param label="γ" value={gamma} set={setGamma} min={0.01} max={0.99} step={0.01} fixed={2} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">训练</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-2">
                <Button onClick={runEpisode} className="flex-1">
                  训练 1 回合
                </Button>
                <Button onClick={reset} variant="outline" className="flex-1">
                  重置
                </Button>
              </div>
              <div className="text-sm text-gray-700 space-y-1">
                <div>已训练回合：{episodeCount}</div>
                <div>θ₀ = {theta0.toFixed(3)}</div>
                <div>θ₁ = {theta1.toFixed(3)}</div>
              </div>
            </CardContent>
          </Card>
          {history.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">回放控制</CardTitle>
              </CardHeader>
              <CardContent>
                <AlgorithmPlayer maxStep={history.length - 1} currentStep={step} onStepChange={setStep} />
              </CardContent>
            </Card>
          )}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">说明</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <p>状态 s∈[-1,1]，动作 a∈[-1,1]，目标是把状态推向 0。</p>
              <p>Critic 用线性 Q(s,a) 近似；Actor 用 tanh 输出确定性动作。</p>
              <p>确定性 Actor 本身不产生探索；实际 off-policy 算法通常需要额外探索机制。</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}
