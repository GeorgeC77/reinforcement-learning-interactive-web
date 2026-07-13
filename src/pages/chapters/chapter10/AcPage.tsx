import { useMemo, useState } from 'react';
import { Activity, AlertTriangle, RefreshCw, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
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
  DEFAULT_CONFIG,
  ACTION_NAMES,
  type GridWorldConfig,
  type Action,
  solveStateValues,
  computeQValues,
  actionValueToStateValue,
  isTerminal,
  step,
  stochasticTransition,
} from '@/lib/rl/gridworld';
import { mulberry32 } from '@/lib/rl/stochasticApproximation';
import {
  qac,
  a2c,
  offPolicyActorCritic,
  qBasedOffPolicyActorCritic,
  deterministicActorCriticEpisode,
  policyWeightedStateValues,
  effectiveSampleSize,
  computeACMetricSeries,
  sampleTdErrorAtStateAction,
  checkBaselineInvariance,
  baselineExpectationMatrix,
  buildActionIndependentBaseline,
  buildActionDependentBaseline,
  solveStateValuesWithSlip,
  computeQValuesWithSlip,
  type ACUpdateRecord,
  type ACResult,
  type DeterministicACStep,
} from '@/lib/rl/actorCritic';

const H_OPTIONS = [10, 20, 30, 50, 100, 200];

type TabKey = 'overview' | 'qac' | 'baseline' | 'a2c' | 'offpolicy' | 'deterministic';
type TaskType = 'episodic' | 'continuing';

function configFromTask(taskType: TaskType): GridWorldConfig {
  return taskType === 'episodic' ? EPISODIC_PATH_CONFIG : DEFAULT_CONFIG;
}

function configDescription(config: GridWorldConfig): string {
  if (config.taskType === 'episodic') {
    return `episodic path-finding：起点 s${config.startState + 1}，目标 s${config.targetState + 1}，禁止状态 [${config.forbiddenStates.map((s) => s + 1).join(', ')}]，每步 ${config.stepReward}，撞墙/禁止 ${config.forbiddenReward}，到达目标 ${config.targetReward}`;
  }
  return `textbook continuing：起点 s${config.startState + 1}，目标 s${config.targetState + 1}（不终止），禁止状态 [${config.forbiddenStates.map((s) => s + 1).join(', ')}]，目标奖励 ${config.targetReward}，禁止奖励 ${config.forbiddenReward}，折扣 γ=${config.gamma}`;
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

function MetricPanel({ result }: { result: ACResult }) {
  const series = useMemo(() => computeACMetricSeries(result, 10), [result]);
  const last = series.length > 0 ? series[series.length - 1] : null;
  return (
    <div className="space-y-4">
      {result.diverged && (
        <div className="flex items-center gap-2 text-red-600 bg-red-50 p-3 rounded-lg border border-red-200 text-sm">
          <AlertTriangle className="w-4 h-4" />
          数值异常已停止训练（约第 {result.divergenceStep} 步）
          {result.divergenceReason ? `：${result.divergenceReason}` : ''}。
          请减小学习率或检查奖励尺度。
        </div>
      )}
      {result.largeMagnitudeWarning && !result.diverged && (
        <div className="flex items-center gap-2 text-amber-600 bg-amber-50 p-3 rounded-lg border border-amber-200 text-sm">
          <AlertTriangle className="w-4 h-4" />
          第 {result.largeMagnitudeWarning.step} 步出现大数值：{result.largeMagnitudeWarning.reason}。
          当前尚未停止，但建议调小学习率。
        </div>
      )}
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500">最近一回合累计奖励</div>
          <div className="font-mono font-semibold">{last ? last.latestReturn.toFixed(2) : '—'}</div>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500">最近 10 回合平均奖励</div>
          <div className="font-mono font-semibold">{last ? last.movingAverageReturn.toFixed(2) : '—'}</div>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500">全训练平均奖励</div>
          <div className="font-mono font-semibold">{last ? last.overallAverageReturn.toFixed(2) : '—'}</div>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500">成功率</div>
          <div className="font-mono font-semibold">{last ? `${(last.successRate * 100).toFixed(0)}%` : '—'}</div>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500">移动平均回合长度</div>
          <div className="font-mono font-semibold">{last ? last.movingAverageEpisodeLength.toFixed(1) : '—'}</div>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500">全训练平均回合长度</div>
          <div className="font-mono font-semibold">{last ? last.overallAverageEpisodeLength.toFixed(1) : '—'}</div>
        </div>
      </div>

      <LineChart
        data={series as any}
        xKey="episode"
        xLabel="回合"
        yLabel="回报"
        series={[
          { key: 'latestReturn', name: 'Episode Return', color: '#2563eb' },
          { key: 'movingAverageReturn', name: 'Moving Average', color: '#ef4444' },
          { key: 'overallAverageReturn', name: 'Overall Average', color: '#22c55e' },
        ]}
        height={160}
      />
      <LineChart
        data={series as any}
        xKey="episode"
        xLabel="回合"
        yLabel="|TD error|"
        series={[{ key: 'meanAbsoluteTdError', name: 'Mean |TD Error|', color: '#f59e0b' }]}
        height={140}
      />
      <LineChart
        data={series as any}
        xKey="episode"
        xLabel="回合"
        yLabel="更新幅度"
        series={[
          { key: 'actorUpdateNorm', name: 'Actor Update Norm', color: '#2563eb' },
          { key: 'criticUpdateNorm', name: 'Critic Update Norm', color: '#22c55e' },
        ]}
        height={140}
      />
      <LineChart
        data={series as any}
        xKey="episode"
        xLabel="回合"
        yLabel="策略变化"
        series={[
          { key: 'meanKL', name: '相邻策略平均 KL 变化', color: '#8b5cf6' },
          { key: 'entropy', name: '策略熵', color: '#06b6d4' },
        ]}
        height={120}
      />
    </div>
  );
}

function TransitionCard({ record }: { record: ACUpdateRecord }) {
  let statusLabel = 'ordinary transition';
  let statusClass = 'bg-gray-100 text-gray-700';
  if (record.done) {
    statusLabel = 'natural terminal';
    statusClass = 'bg-green-100 text-green-700';
  } else if (record.truncated) {
    statusLabel = 'horizon truncated';
    statusClass = 'bg-amber-100 text-amber-700';
  }

  return (
    <div className="space-y-2">
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
      <div className={`inline-block px-2 py-1 rounded text-xs ${statusClass}`}>{statusLabel}</div>
      {record.bootstrapUsed !== undefined && (
        <div
          className={`inline-block px-2 py-1 rounded text-xs ml-2 ${
            record.bootstrapUsed ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
          }`}
        >
          {record.bootstrapUsed ? 'bootstrap used' : 'no bootstrap'}
        </div>
      )}
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
  bootstrapOnTruncation,
  setBootstrapOnTruncation,
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
  bootstrapOnTruncation?: boolean;
  setBootstrapOnTruncation?: (v: boolean) => void;
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
              <SelectItem value="episodic">episodic path-finding</SelectItem>
              <SelectItem value="continuing">textbook continuing</SelectItem>
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
      {bootstrapOnTruncation !== undefined && setBootstrapOnTruncation && (
        <div className="flex items-center justify-between rounded-lg border border-gray-200 p-3">
          <div className="space-y-0.5">
            <label className="text-sm font-medium text-gray-700">horizon truncation 时 bootstrap</label>
            <p className="text-xs text-gray-500">关闭后把人为截断视为终止，避免时间限制偏差。</p>
          </div>
          <Switch
            checked={bootstrapOnTruncation}
            onCheckedChange={setBootstrapOnTruncation}
          />
        </div>
      )}
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
  const config = useMemo(() => EPISODIC_PATH_CONFIG, []);
  const result = useMemo(
    () =>
      a2c(config, {
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
            description="这里的 Critic 估计状态值 V；δ_t 是 advantage A_π(S_t,A_t) 的单步样本估计。QAC（用 Q 直接驱动 Actor）在下一 tab 单独介绍。"
          />
          {record && (
            <>
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
              <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
                <GridWorld
                  config={config}
                  policy={record.actorFullPolicyAfter ?? Array.from({ length: config.rows * config.cols }, () => Array.from({ length: 5 }, () => 1 / 5))}
                  values={record.vAfter ?? new Array(config.rows * config.cols).fill(0)}
                  showValues
                  highlightState={record.state}
                  highlightNextState={record.nextState}
                  highlightAction={{ state: record.state, action: record.action }}
                  highlightUpdatedState={record.state}
                  className="max-w-full"
                />
                <div className="mt-2 flex flex-wrap gap-3 text-xs text-gray-600">
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#3a7bd5]" /> current S_t</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#f59e0b]" /> next S_{'{t+1}'}</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#22c55e]" /> updated state</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#1a3a5c]" /> current action</span>
                  {record.done && <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-green-200" /> natural terminal</span>}
                  {record.truncated && <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-amber-200" /> horizon truncated</span>}
                </div>
              </div>
            </>
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
  const [actorWeightMode, setActorWeightMode] = useState<'raw-q' | 'advantage'>('raw-q');
  const [bootstrapOnTruncation, setBootstrapOnTruncation] = useState(true);
  const [qShift, setQShift] = useState(0);

  const config = useMemo(() => configFromTask(taskType), [taskType]);
  const result = useMemo(
    () =>
      qac(config, {
        seed,
        horizonH,
        actorAlpha,
        criticAlpha,
        episodes,
        actorWeightMode,
        bootstrapOnTruncation,
      }),
    [config, seed, horizonH, actorAlpha, criticAlpha, episodes, actorWeightMode, bootstrapOnTruncation]
  );
  const [step, setStep] = useState(0);
  const record = result.updates[Math.min(step, result.updates.length - 1)];

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
                <GridWorld
                  config={config}
                  policy={currentPolicy}
                  values={values}
                  showValues
                  highlightState={record.state}
                  highlightNextState={record.nextState}
                  highlightAction={{ state: record.state, action: record.action }}
                  highlightUpdatedState={record.state}
                  className="max-w-full"
                />
                <div className="mt-2 flex flex-wrap gap-3 text-xs text-gray-600">
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#3a7bd5]" /> current S_t</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#f59e0b]" /> next S_{'{t+1}'}</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#22c55e]" /> updated state</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#1a3a5c]" /> current action</span>
                  {record.done && <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-green-200" /> natural terminal</span>}
                  {record.truncated && <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-amber-200" /> horizon truncated</span>}
                </div>
                <p className="mt-1 text-xs text-gray-600">
                  {valueMode === 'actor'
                    ? 'V_π(s) = Σ_a π(a|s) q(s,a)，对应当前 Actor 的随机策略。'
                    : 'max_a q(s,a)，对应贪婪策略的值。'}
                </p>
              </div>
            </>
          )}
          <MetricPanel result={result} />
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
                bootstrapOnTruncation={bootstrapOnTruncation}
                setBootstrapOnTruncation={setBootstrapOnTruncation}
              >
                <div>
                  <label className="text-sm text-gray-700 block mb-1">Actor 权重模式</label>
                  <Select value={actorWeightMode} onValueChange={(v) => setActorWeightMode(v as 'raw-q' | 'advantage')}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="raw-q">raw q(s,a)</SelectItem>
                      <SelectItem value="advantage">q(s,a) - V_π(s)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </DiscreteControlPanel>
              <div className="mt-3 text-xs text-gray-600">{configDescription(config)}</div>
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
              <CardTitle className="text-base">Q 平移实验</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-gray-700">
              <p>对当前状态 s 的所有动作 Q(s,·) 同时加上常数 c(s)：</p>
              <Param label="平移量 c(s)" value={qShift} set={setQShift} min={-5} max={5} step={0.5} fixed={1} />
              {record && (
                <div className="space-y-2">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-gray-50 rounded p-2">
                      <div className="text-gray-500">raw-Q weight</div>
                      <div className="font-mono">{(record.criticEstimateBefore + qShift).toFixed(4)}</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2">
                      <div className="text-gray-500">advantage weight</div>
                      <div className="font-mono">
                        {(
                          record.criticEstimateBefore -
                          record.actorPolicyBefore.reduce((sum, p, a) => sum + p * (record.qBefore![record.state][a] + qShift), 0)
                        ).toFixed(4)}
                      </div>
                    </div>
                  </div>
                  <p className="text-xs text-gray-600">
                    平移后 raw-Q 的 Actor 权重改变，但 advantage 权重不变，因为 V_π(s) 也平移了相同的量。
                  </p>
                </div>
              )}
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
  const config = useMemo(() => EPISODIC_PATH_CONFIG, []);
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
  const numStates = config.rows * config.cols;
  const uniformPolicy = useMemo(
    () => Array.from({ length: numStates }, () => Array.from({ length: 5 }, () => 1 / 5)),
    [numStates]
  );
  const baselineV = useMemo(() => solveStateValues(uniformPolicy, config), [uniformPolicy, config]);
  const baselineQ = useMemo(() => computeQValues(baselineV, config), [baselineV, config]);
  const independentBaseline = useMemo(() => buildActionIndependentBaseline(numStates, 1.0), [numStates]);
  const dependentBaseline = useMemo(() => buildActionDependentBaseline(baselineQ), [baselineQ]);
  const independentCheck = useMemo(
    () => checkBaselineInvariance(uniformPolicy, independentBaseline),
    [uniformPolicy, independentBaseline]
  );
  const dependentCheck = useMemo(() => checkBaselineInvariance(uniformPolicy, dependentBaseline), [uniformPolicy, dependentBaseline]);
  const [baselineState, setBaselineState] = useState(0);
  const independentMatrix = useMemo(
    () =>
      baselineExpectationMatrix({
        policy: uniformPolicy[baselineState],
        state: baselineState,
        baselineByAction: independentBaseline[baselineState],
      }),
    [uniformPolicy, baselineState, independentBaseline]
  );
  const dependentMatrix = useMemo(
    () =>
      baselineExpectationMatrix({
        policy: uniformPolicy[baselineState],
        state: baselineState,
        baselineByAction: dependentBaseline[baselineState],
      }),
    [uniformPolicy, baselineState, dependentBaseline]
  );

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
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Baseline 不变性数值验证</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-green-50 rounded p-2 border border-green-100">
                  <div className="text-green-700 text-xs">action-independent b(s)=1</div>
                  <div className="font-mono">max |Σ_a π score·b| = {independentCheck.maxAbs.toExponential(2)}</div>
                  <div className="text-xs text-green-700 mt-1">
                    {independentCheck.isInvariant ? '≈ 0，符合不变性' : '非零，请检查实现'}
                  </div>
                </div>
                <div className="bg-red-50 rounded p-2 border border-red-100">
                  <div className="text-red-700 text-xs">action-dependent b(s,a)=Q(s,a)</div>
                  <div className="font-mono">max |Σ_a π score·b| = {dependentCheck.maxAbs.toFixed(4)}</div>
                  <div className="text-xs text-red-700 mt-1">通常非零，会改变策略梯度</div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-700">查看状态</label>
                <Select value={String(baselineState)} onValueChange={(v) => setBaselineState(Number(v))}>
                  <SelectTrigger className="w-32">
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

              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <div className="text-xs text-green-700 mb-1">action-independent: π(a|s) b(s) ∇logπ(a|s)</div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs border-collapse">
                      <thead>
                        <tr className="text-gray-500 border-b">
                          <th className="text-left py-1">a</th>
                          {independentMatrix[0]?.map((_, k) => (
                            <th key={k} className="text-right py-1">k={k}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {independentMatrix.map((row, a) => (
                          <tr key={a} className="border-b border-gray-100">
                            <td className="py-1 font-mono">{ACTION_NAMES[a]}</td>
                            {row.map((v, k) => (
                              <td key={k} className="text-right py-1 font-mono">
                                {Math.abs(v) < 1e-9 ? '0' : v.toExponential(1)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                <div>
                  <div className="text-xs text-red-700 mb-1">action-dependent: π(a|s) Q(s,a) ∇logπ(a|s)</div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs border-collapse">
                      <thead>
                        <tr className="text-gray-500 border-b">
                          <th className="text-left py-1">a</th>
                          {dependentMatrix[0]?.map((_, k) => (
                            <th key={k} className="text-right py-1">k={k}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {dependentMatrix.map((row, a) => (
                          <tr key={a} className="border-b border-gray-100">
                            <td className="py-1 font-mono">{ACTION_NAMES[a]}</td>
                            {row.map((v, k) => (
                              <td key={k} className="text-right py-1 font-mono">
                                {Math.abs(v) < 1e-9 ? '0' : v.toExponential(1)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              <div className="border-t pt-3">
                <div className="text-xs text-gray-700 mb-1">
                  全状态参数分量求和 Σ_a π(a|s) score(s,a)[k] b(s,a)
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr className="text-gray-500 border-b">
                        <th className="text-left py-1">s</th>
                        {independentCheck.perStateComponentSum[0]?.map((_, k) => (
                          <th key={k} className="text-right py-1">k={k}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {independentCheck.perStateComponentSum.map((indepRow, s) => {
                        const depRow = dependentCheck.perStateComponentSum[s];
                        return (
                          <tr key={s} className="border-b border-gray-100">
                            <td className="py-1 font-mono">s{s + 1}</td>
                            {indepRow.map((iv, k) => {
                              const dv = depRow?.[k] ?? 0;
                              const isZero = Math.abs(iv) < 1e-9;
                              const isNonZero = Math.abs(dv) > 1e-9;
                              return (
                                <td key={k} className="text-right py-1 font-mono">
                                  <span className={isZero ? 'text-green-700' : ''}>
                                    {isZero ? '0' : iv.toExponential(1)}
                                  </span>
                                  {' / '}
                                  <span className={isNonZero ? 'text-red-700' : ''}>
                                    {isNonZero ? dv.toExponential(1) : '0'}
                                  </span>
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="flex gap-4 text-xs mt-1">
                  <span className="text-green-700">绿色：action-independent baseline</span>
                  <span className="text-red-700">红色：action-dependent baseline</span>
                </div>
              </div>

              <p className="text-xs text-gray-600">
                矩阵每一行对应一个动作 a，每一列对应一个策略参数分量 k。对 softmax 策略，action-independent baseline 的列和（按动作求和）恒为 0；
                action-dependent baseline（如 Q）的列和通常不为 0。
              </p>
            </CardContent>
          </Card>
          <MetricPanel result={result} />
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">说明</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <p>
                动作无关的 baseline 不改变期望策略梯度。
                合适的状态价值 baseline 和 bootstrap 通常可以降低方差，
                但 Critic 不准确时会引入估计误差，方差下降并非无条件保证。
              </p>
              <p>
                1. 若 baseline 与动作无关，b(s) 可提到求和号外：
                <KaTeX math={String.raw`\sum_a \pi(a|s) \nabla\log\pi(a|s) b(s) = b(s) \sum_a \pi(a|s) \nabla\log\pi(a|s) = 0`} display={false} />
                。
              </p>
              <p>
                2. 若 baseline 与动作相关，b(s,a) 不能提出，期望项通常不为 0：
                <KaTeX math={String.raw`\sum_a \pi(a|s) \nabla\log\pi(a|s) b(s,a) \neq 0`} display={false} />
                。
              </p>
              <p>
                下面矩阵显示每个状态、每个策略参数分量的求和结果。
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

function sampleFromDist(dist: { nextState: number; prob: number }[], rng: () => number): number {
  const r = rng();
  let cum = 0;
  for (const { nextState, prob } of dist) {
    cum += prob;
    if (r <= cum) return nextState;
  }
  return dist[dist.length - 1]?.nextState ?? 0;
}

function ExactAdvantagePanel({ config }: { config: GridWorldConfig }) {
  const numStates = config.rows * config.cols;
  const nonTerminalStates = useMemo(
    () => Array.from({ length: numStates }, (_, s) => s).filter((s) => !isTerminal(s, config)),
    [numStates, config]
  );
  const [state, setState] = useState(0);
  const [action, setAction] = useState<Action>(1);
  const [slip, setSlip] = useState(0);

  const uniformPolicy = useMemo(
    () => Array.from({ length: numStates }, () => Array.from({ length: 5 }, () => 1 / 5)),
    [numStates]
  );

  const { trueV, trueAdvantage } = useMemo(() => {
    if (slip === 0) {
      const v = solveStateValues(uniformPolicy, config);
      const q = computeQValues(v, config);
      return { trueV: v, trueAdvantage: q[state]?.[action] - v[state] };
    }
    const v = solveStateValuesWithSlip(uniformPolicy, config, slip);
    const q = computeQValuesWithSlip(uniformPolicy, config, slip);
    return { trueV: v, trueAdvantage: q[state]?.[action] - v[state] };
  }, [uniformPolicy, config, state, action, slip]);

  const estimateResult = useMemo(() => {
    try {
      if (slip === 0) {
        const rng = mulberry32(42);
        const est = sampleTdErrorAtStateAction({ config, state, action, values: trueV, rng });
        if (est.count === 0) {
          return { ok: false as const, error: 'Cannot sample TD error: state is terminal or no valid transition available.' };
        }
        return { ok: true as const, est, deterministic: true as const };
      }

      const rng = mulberry32(42);
      const samples: number[] = [];
      for (let i = 0; i < 500; i++) {
        const dist = stochasticTransition(state, action, config, slip);
        const nextState = sampleFromDist(dist, rng);
        const { reward, done } = step(state, action, config);
        const bootstrap = done ? 0 : trueV[nextState];
        const tdError = reward + config.gamma * bootstrap - trueV[state];
        samples.push(tdError);
      }
      const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
      const variance = samples.reduce((a, b) => a + (b - mean) * (b - mean), 0) / samples.length;
      const std = Math.sqrt(variance);
      const se = std / Math.sqrt(samples.length);
      const ciLower = mean - 1.96 * se;
      const ciUpper = mean + 1.96 * se;
      return {
        ok: true as const,
        est: { samples, mean, std, count: samples.length },
        deterministic: false as const,
        se,
        ciLower,
        ciUpper,
      };
    } catch (e) {
      return { ok: false as const, error: e instanceof Error ? e.message : String(e) };
    }
  }, [config, state, action, trueV, slip]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">精确对照：TD error 的期望 vs 真实 Advantage</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <div>
            <label className="text-sm text-gray-700 block mb-1">状态</label>
            <Select value={String(state)} onValueChange={(v) => setState(Number(v))}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {nonTerminalStates.map((s) => (
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
          <div>
            <label className="text-sm text-gray-700 block mb-1">随机滑移概率</label>
            <Select value={String(slip)} onValueChange={(v) => setSlip(Number(v))}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0">0（确定性）</SelectItem>
                <SelectItem value="0.1">0.1</SelectItem>
                <SelectItem value="0.2">0.2</SelectItem>
                <SelectItem value="0.3">0.3</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        {estimateResult.ok ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-gray-50 rounded p-2">
                <div className="text-gray-500 text-xs">真实 A_π(s,a)</div>
                <div className="font-mono">{trueAdvantage.toFixed(4)}</div>
              </div>
              <div className="bg-gray-50 rounded p-2">
                <div className="text-gray-500 text-xs">样本 δ 均值</div>
                <div className="font-mono">{estimateResult.est.mean.toFixed(4)}</div>
              </div>
              <div className="bg-gray-50 rounded p-2">
                <div className="text-gray-500 text-xs">样本 δ 标准差</div>
                <div className="font-mono">{estimateResult.est.std.toFixed(4)}</div>
              </div>
              <div className="bg-gray-50 rounded p-2">
                <div className="text-gray-500 text-xs">绝对误差</div>
                <div className="font-mono">
                  {Math.abs(estimateResult.est.mean - trueAdvantage).toFixed(4)}
                </div>
              </div>
            </div>
            {'deterministic' in estimateResult && !estimateResult.deterministic && (
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-gray-50 rounded p-2">
                  <div className="text-gray-500 text-xs">标准误 SE</div>
                  <div className="font-mono">{estimateResult.se.toFixed(4)}</div>
                </div>
                <div className="bg-gray-50 rounded p-2">
                  <div className="text-gray-500 text-xs">95% CI 下限</div>
                  <div className="font-mono">{estimateResult.ciLower.toFixed(4)}</div>
                </div>
                <div className="bg-gray-50 rounded p-2">
                  <div className="text-gray-500 text-xs">95% CI 上限</div>
                  <div className="font-mono">{estimateResult.ciUpper.toFixed(4)}</div>
                </div>
              </div>
            )}
            <div className="text-xs text-gray-600">sample count: {estimateResult.est.count}</div>
          </>
        ) : (
          <div className="text-red-600 bg-red-50 p-3 rounded-lg border border-red-200 text-sm">
            {estimateResult.error}
          </div>
        )}
        <p className="text-xs text-gray-600">
          {slip === 0
            ? '确定性环境：标准差为 0，count=1 的样本均值即为真实 advantage。'
            : '随机滑移环境：运行 500 个种子得到 Monte Carlo 估计，标准误与置信区间衡量估计精度。'}
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
  const [bootstrapOnTruncation, setBootstrapOnTruncation] = useState(true);

  const config = useMemo(() => configFromTask(taskType), [taskType]);
  const result = useMemo(
    () =>
      a2c(config, {
        seed,
        horizonH,
        actorAlpha,
        criticAlpha,
        episodes,
        bootstrapOnTruncation,
      }),
    [config, seed, horizonH, actorAlpha, criticAlpha, episodes, bootstrapOnTruncation]
  );
  const [step, setStep] = useState(0);
  const record = result.updates[Math.min(step, result.updates.length - 1)];

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
                <GridWorld
                  config={config}
                  policy={policy}
                  values={values}
                  showValues
                  highlightState={record.state}
                  highlightNextState={record.nextState}
                  highlightAction={{ state: record.state, action: record.action }}
                  highlightUpdatedState={record.state}
                  className="max-w-full"
                />
                <div className="mt-2 flex flex-wrap gap-3 text-xs text-gray-600">
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#3a7bd5]" /> current S_t</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#f59e0b]" /> next S_{'{t+1}'}</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#22c55e]" /> updated state</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#1a3a5c]" /> current action</span>
                  {record.done && <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-green-200" /> natural terminal</span>}
                  {record.truncated && <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-amber-200" /> horizon truncated</span>}
                </div>
              </div>
            </>
          )}
          <MetricPanel result={result} />
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
                bootstrapOnTruncation={bootstrapOnTruncation}
                setBootstrapOnTruncation={setBootstrapOnTruncation}
              />
              <div className="mt-3 text-xs text-gray-600">{configDescription(config)}</div>
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
  const [bootstrapOnTruncation, setBootstrapOnTruncation] = useState(true);
  const [importanceMode, setImportanceMode] = useState<'raw' | 'clipped'>('raw');
  const [clipThreshold, setClipThreshold] = useState(10);

  const config = useMemo(() => configFromTask(taskType), [taskType]);
  const result = useMemo(() => {
    const options = {
      seed,
      horizonH,
      actorAlpha,
      criticAlpha,
      episodes,
      epsilon,
      bootstrapOnTruncation,
      importanceMode,
      clipThreshold,
    };
    try {
      return variant === 'textbook'
        ? offPolicyActorCritic(config, options)
        : qBasedOffPolicyActorCritic(config, options);
    } catch (err) {
      return null;
    }
  }, [config, seed, horizonH, actorAlpha, criticAlpha, episodes, epsilon, variant, bootstrapOnTruncation, importanceMode, clipThreshold]);

  const [step, setStep] = useState(0);
  const record = result ? result.updates[Math.min(step, result.updates.length - 1)] : undefined;

  const values = useMemo(() => {
    if (!record) return new Array(config.rows * config.cols).fill(0);
    if (variant === 'textbook' && record.vAfter) return record.vAfter;
    if (variant === 'extended' && record.qAfter) return actionValueToStateValue(record.qAfter);
    return new Array(config.rows * config.cols).fill(0);
  }, [record, variant, config.rows, config.cols]);

  const policy = record?.actorFullPolicyAfter ?? Array.from({ length: config.rows * config.cols }, () => Array.from({ length: 5 }, () => 1 / 5));

  const rhoStats = useMemo(() => {
    if (!result) return null;
    const rawRhos = result.updates.map((u) => u.rawRho ?? u.rho ?? 1);
    const usedRhos = result.updates.map((u) => u.usedRho ?? u.rho ?? 1);
    const maxRaw = Math.max(...rawRhos);
    const maxUsed = Math.max(...usedRhos);
    const meanRaw = rawRhos.reduce((a, b) => a + b, 0) / rawRhos.length;
    const meanUsed = usedRhos.reduce((a, b) => a + b, 0) / usedRhos.length;
    const clippedCount = result.updates.filter((u) => u.wasClipped).length;
    const clippedRatio = clippedCount / result.updates.length;
    const essRaw = effectiveSampleSize(rawRhos);
    const essUsed = effectiveSampleSize(usedRhos);
    return { maxRaw, maxUsed, meanRaw, meanUsed, clippedRatio, essRaw, essUsed, rawRhos, usedRhos };
  }, [result]);

  const histogram = useMemo(() => {
    if (!rhoStats) return null;
    const bins = [0, 1, 2, 5, 10, 20, 50, 100];
    const rawCounts = new Array(bins.length).fill(0);
    const usedCounts = new Array(bins.length).fill(0);
    for (const r of rhoStats.rawRhos) {
      const idx = bins.findIndex((_, i) => (bins[i + 1] === undefined ? true : r < bins[i + 1]));
      rawCounts[idx]++;
    }
    for (const r of rhoStats.usedRhos) {
      const idx = bins.findIndex((_, i) => (bins[i + 1] === undefined ? true : r < bins[i + 1]));
      usedCounts[idx]++;
    }
    return { bins, rawCounts, usedCounts };
  }, [rhoStats]);

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
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div className="bg-gray-50 rounded p-2">
                      <div className="text-gray-500 text-xs">target π(A|S)</div>
                      <div className="font-mono">{record.targetProb?.toFixed(4)}</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2">
                      <div className="text-gray-500 text-xs">behavior β(A|S)</div>
                      <div className="font-mono">{record.behaviorProb?.toFixed(4)}</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2">
                      <div className="text-gray-500 text-xs">raw ρ</div>
                      <div className="font-mono">{record.rawRho?.toFixed(4) ?? record.rho?.toFixed(4)}</div>
                    </div>
                    <div className="bg-blue-50 rounded p-2 border border-blue-100">
                      <div className="text-blue-700 text-xs">used ρ</div>
                      <div className="font-mono font-semibold">
                        {record.usedRho?.toFixed(4) ?? record.rho?.toFixed(4)}
                        {record.wasClipped && <span className="ml-1 text-amber-600 text-xs">(clipped)</span>}
                      </div>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <CriticCard record={record} />
                    <ActorCard record={record} />
                  </div>
                </CardContent>
              </Card>
              <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
                <GridWorld
                  config={config}
                  policy={policy}
                  values={values}
                  showValues
                  highlightState={record.state}
                  highlightNextState={record.nextState}
                  highlightAction={{ state: record.state, action: record.action }}
                  highlightUpdatedState={record.state}
                  className="max-w-full"
                />
                <div className="mt-2 flex flex-wrap gap-3 text-xs text-gray-600">
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#3a7bd5]" /> current S_t</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#f59e0b]" /> next S_{'{t+1}'}</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#22c55e]" /> updated state</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-[#1a3a5c]" /> current action</span>
                  {record.done && <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-green-200" /> natural terminal</span>}
                  {record.truncated && <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-amber-200" /> horizon truncated</span>}
                </div>
              </div>
              {rhoStats && (
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">ρ 统计</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div className="bg-gray-50 rounded p-2">
                        <div className="text-gray-500 text-xs">max raw ρ</div>
                        <div className="font-mono">{rhoStats.maxRaw.toFixed(2)}</div>
                      </div>
                      <div className="bg-gray-50 rounded p-2">
                        <div className="text-gray-500 text-xs">max used ρ</div>
                        <div className="font-mono">{rhoStats.maxUsed.toFixed(2)}</div>
                      </div>
                      <div className="bg-gray-50 rounded p-2">
                        <div className="text-gray-500 text-xs">clip 比例</div>
                        <div className="font-mono">{(rhoStats.clippedRatio * 100).toFixed(1)}%</div>
                      </div>
                      <div className="bg-gray-50 rounded p-2">
                        <div className="text-gray-500 text-xs">ESS used</div>
                        <div className="font-mono">{rhoStats.essUsed.toFixed(2)}</div>
                      </div>
                    </div>
                    {histogram && (
                      <div className="space-y-2">
                        <div className="text-xs text-gray-500">ρ 分布直方图（raw / used）</div>
                        <div className="space-y-1">
                          {histogram.bins.map((bin, i) => {
                            const rawCount = histogram.rawCounts[i];
                            const usedCount = histogram.usedCounts[i];
                            const maxCount = Math.max(...histogram.rawCounts, ...histogram.usedCounts, 1);
                            const label =
                              i === histogram.bins.length - 1 ? `≥ ${bin}` : `[${bin}, ${histogram.bins[i + 1]})`;
                            return (
                              <div key={i} className="flex items-center gap-2 text-xs">
                                <span className="w-16 text-gray-500">{label}</span>
                                <div className="flex-1 h-4 bg-gray-100 rounded overflow-hidden relative">
                                  <div
                                    className="absolute top-0 left-0 h-full bg-blue-400"
                                    style={{ width: `${(rawCount / maxCount) * 100}%` }}
                                  />
                                  <div
                                    className="absolute top-0 left-0 h-full bg-amber-400 opacity-70"
                                    style={{ width: `${(usedCount / maxCount) * 100}%` }}
                                  />
                                </div>
                                <span className="w-20 text-right font-mono">{rawCount}/{usedCount}</span>
                              </div>
                            );
                          })}
                        </div>
                        <p className="text-xs text-gray-600">
                          蓝色为 raw ρ，琥珀色为 used ρ。Clipping 降低方差但引入偏差，ESS 反映有效样本量。
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </>
          )}
          {result && <MetricPanel result={result} />}
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
                bootstrapOnTruncation={bootstrapOnTruncation}
                setBootstrapOnTruncation={setBootstrapOnTruncation}
              >
                <Param label="行为策略 ε" value={epsilon} set={setEpsilon} min={0.05} max={1} step={0.05} fixed={2} />
                <div>
                  <label className="text-sm text-gray-700 block mb-1">重要性采样模式</label>
                  <Select value={importanceMode} onValueChange={(v) => setImportanceMode(v as 'raw' | 'clipped')}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="raw">raw（无偏，方差大）</SelectItem>
                      <SelectItem value="clipped">clipped（有偏，方差小）</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                {importanceMode === 'clipped' && (
                  <Param label="clip 阈值" value={clipThreshold} set={setClipThreshold} min={1} max={100} step={1} fixed={0} />
                )}
              </DiscreteControlPanel>
              <div className="mt-3 text-xs text-gray-600">{configDescription(config)}</div>
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
  const [diverged, setDiverged] = useState(false);
  const [divergenceStep, setDivergenceStep] = useState<number | undefined>();
  const [divergenceReason, setDivergenceReason] = useState<string | undefined>();
  const [largeMagnitudeWarning, setLargeMagnitudeWarning] = useState<{ step: number; reason: string } | undefined>();

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
    setDiverged(res.diverged);
    setDivergenceStep(res.divergenceStep);
    setDivergenceReason(res.divergenceReason);
    setLargeMagnitudeWarning(res.largeMagnitudeWarning);
  }

  function reset() {
    setTheta0(0.5);
    setTheta1(0.0);
    setW([0, 0, 0, 0, 0, 0]);
    setEpisodeCount(0);
    setHistory([]);
    setStep(0);
    setDiverged(false);
    setDivergenceStep(undefined);
    setDivergenceReason(undefined);
    setLargeMagnitudeWarning(undefined);
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

          {diverged && (
            <div className="flex items-center gap-2 text-red-600 bg-red-50 p-3 rounded-lg border border-red-200 text-sm">
              <AlertTriangle className="w-4 h-4" />
              数值异常已停止（约第 {divergenceStep} 步）
              {divergenceReason ? `：${divergenceReason}` : ''}。
            </div>
          )}
          {largeMagnitudeWarning && !diverged && (
            <div className="flex items-center gap-2 text-amber-600 bg-amber-50 p-3 rounded-lg border border-amber-200 text-sm">
              <AlertTriangle className="w-4 h-4" />
              第 {largeMagnitudeWarning.step} 步出现大数值：{largeMagnitudeWarning.reason}。
            </div>
          )}
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
                    <div className="text-gray-500 text-xs">μ(s) actor action</div>
                    <div className="font-mono">{currentStep.actorAction.toFixed(3)}</div>
                  </div>
                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-gray-500 text-xs">behavior action</div>
                    <div className="font-mono">{currentStep.behaviorAction.toFixed(3)}</div>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-gray-500 text-xs">exploration noise</div>
                    <div className="font-mono">{currentStep.explorationNoise.toFixed(4)}</div>
                  </div>
                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-gray-500 text-xs">r</div>
                    <div className="font-mono">{currentStep.reward.toFixed(4)}</div>
                  </div>
                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-gray-500 text-xs">distance to target</div>
                    <div className="font-mono">{Math.abs(currentStep.state).toFixed(4)}</div>
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
                    <div className="text-gray-500 text-xs">critic loss ½δ²</div>
                    <div className="font-mono">{currentStep.criticLoss.toFixed(4)}</div>
                  </div>
                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-gray-500 text-xs">∂Q/∂a at a=μ(s)</div>
                    <div className="font-mono">{currentStep.dqda.toFixed(4)}</div>
                  </div>
                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-gray-500 text-xs">Critic weights for Actor</div>
                    <div className="font-mono">{currentStep.criticWeightsUsedByActor}</div>
                  </div>
                </div>
                <div className="bg-gray-50 rounded p-2">
                  <div className="text-gray-500 text-xs">actor gradient</div>
                  <div className="font-mono">
                    [{currentStep.actorGradient.map((g) => g.toFixed(4)).join(', ')}]
                  </div>
                </div>
                <p className="text-xs text-gray-600">
                  μ_θ(s) + exploration noise = executed action。Actor 梯度在 a = μ_θ(s) 处计算；Critic 使用实际执行的 behaviorAction 更新。
                </p>
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
