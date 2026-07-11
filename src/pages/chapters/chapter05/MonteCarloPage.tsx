import { useState, useCallback, useMemo, useRef } from 'react';
import { Dices, ShieldAlert } from 'lucide-react';
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
import {
  DEFAULT_CONFIG,
  ACTION_NAMES,
  type Action,
  type Policy,
  epsilonGreedyPolicy,
  greedyPolicy,
  actionValueToStateValue,
  mcBasic,
  type MCBasicIteration,
  mcBasicPolicyIteration,
  mcExploringStarts,
  mcEpsilonGreedy,
  createMCLearnerState,
  runMCExploringStartsEpisodes,
  runMCEpsilonGreedyEpisodes,
  type MCLearnerState,
  type EpsilonSchedule,
  estimateTrueActionValues,
  solveStateValues,
  computeQValues,
  qTableRMSE,
} from '@/lib/rl/gridworld';

type TabKey = 'basic' | 'exploring' | 'epsilon';
type VisitMode = 'first-visit' | 'every-visit';

interface LearningPoint {
  episode: number;
  rmse: number;
}

export default function Chapter05MonteCarloPage() {
  const [activeTab, setActiveTab] = useState<TabKey>('basic');

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Dices className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          第 5 章 蒙特卡洛方法
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          蒙特卡洛方法不依赖模型，通过采样完整回合并用样本回报估计动作值。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="蒙特卡洛回报"
          formula={<KaTeX math={String.raw`G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots + \gamma^{T-t-1} r_T`} display />}
          description="从时刻 t 开始，把折扣后的未来奖励相加得到一个样本回报。"
        />
        <FormulaCard
          title="样本平均更新"
          formula={<KaTeX math={String.raw`q(s,a) \leftarrow q(s,a) + \frac{1}{N(s,a)} \bigl(G - q(s,a)\bigr)`} display />}
          description="每遇到一次 (s,a)，用新样本逐步修正估计。"
        />
        <FormulaCard
          title="ε-贪心策略"
          formula={
            <KaTeX
              math={String.raw`\pi(a|s) = \begin{cases} 1-\varepsilon + \frac{\varepsilon}{|\mathcal{A}|}, & a = \arg\max_{a'} q(s,a') \\ \frac{\varepsilon}{|\mathcal{A}|}, & \text{否则} \end{cases}`}
              display
            />
          }
          description="大部分时候选择当前最优动作，偶尔随机探索。"
        />
      </section>

      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as TabKey)}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="basic">MC Basic</TabsTrigger>
          <TabsTrigger value="exploring">Exploring Starts</TabsTrigger>
          <TabsTrigger value="epsilon">ε-Greedy</TabsTrigger>
        </TabsList>

        <TabsContent value="basic" className="mt-4">
          <MCBasicDemo />
        </TabsContent>
        <TabsContent value="exploring" className="mt-4">
          <MCExploringStartsDemo />
        </TabsContent>
        <TabsContent value="epsilon" className="mt-4">
          <MCEpsilonGreedyDemo />
        </TabsContent>
      </Tabs>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">观察与思考</h2>
        <ul className="space-y-3 text-gray-700">
          <li>
            <strong>MC Basic：</strong>完整的策略迭代算法（算法 5.1）。每轮对每对 (s,a) 采样若干回合估计 q_π，然后做贪心策略改进，直到策略稳定。
          </li>
          <li>
            <strong>Exploring Starts：</strong>用随机初始 (s,a) 保证每对动作值都能被访问，随后遵循当前贪心策略。这是从策略迭代改造而来的 MC 控制。
          </li>
          <li>
            <strong>ε-Greedy：</strong>当无法保证 exploring starts 时，用 ε-贪心持续探索，是更实用的模型-free 控制。支持固定 ε 和衰减 ε 两种调度。
          </li>
          <li>
            <strong>First-visit vs Every-visit：</strong>First-visit 只使用每个回合中首次访问 (s,a) 后的回报；
            Every-visit 使用该回合中的全部访问，因此能利用更多样本，但同一回合内
            这些回报通常相关。两种方法在适当条件下均可收敛，其有限样本偏差与方差
            取决于具体采样过程。
          </li>
        </ul>
      </section>
    </div>
  );
}

// ------------------- MC Basic -------------------
function MCBasicDemo() {
  const config = DEFAULT_CONFIG;
  const [episodesPerPair, setEpisodesPerPair] = useState(20);
  const [mode, setMode] = useState<'single-eval' | 'policy-iteration'>('policy-iteration');
  const [result, setResult] = useState<
    | { type: 'single-eval'; q: number[][]; counts: number[][]; rmse: number; policyName: string }
    | { type: 'policy-iteration'; iterations: MCBasicIteration[]; finalPolicy: Policy; finalQ: number[][]; rmse: number; stable: boolean }
    | null
  >(null);
  const [iterIndex, setIterIndex] = useState(0);

  function run() {
    if (mode === 'single-eval') {
      const { policy, qValues, returns } = mcBasic(config, episodesPerPair, 30);
      const counts = returns.map((s) => s.map((a) => a.length));
      const vTrue = solveStateValues(policy, config);
      const qTrue = computeQValues(vTrue, config);
      const rmse = qTableRMSE(qValues, qTrue);
      setResult({ type: 'single-eval', q: qValues, counts, rmse, policyName: '随机策略（被评估）' });
      setIterIndex(0);
    } else {
      const { iterations, finalPolicy, finalQ } = mcBasicPolicyIteration(
        config,
        episodesPerPair,
        30,
        20
      );
      const vTrue = solveStateValues(finalPolicy, config);
      const qTrue = computeQValues(vTrue, config);
      const rmse = qTableRMSE(finalQ, qTrue);
      setResult({
        type: 'policy-iteration',
        iterations,
        finalPolicy,
        finalQ,
        rmse,
        stable: iterations[iterations.length - 1]?.policyStable ?? false,
      });
      setIterIndex(0);
    }
  }

  const displayPolicy = useMemo(() => {
    if (!result) return null;
    if (result.type === 'single-eval') return greedyPolicy(result.q);
    if (result.type === 'policy-iteration') {
      if (iterIndex === 0 && result.iterations.length > 0) {
        return result.iterations[iterIndex].policyBefore;
      }
      return result.iterations[iterIndex]?.policyAfter ?? result.finalPolicy;
    }
    return null;
  }, [result, iterIndex]);

  const displayValues = useMemo(() => {
    if (!result) return null;
    if (result.type === 'single-eval') return actionValueToStateValue(result.q);
    if (result.type === 'policy-iteration') {
      return actionValueToStateValue(result.iterations[iterIndex]?.qEstimate ?? result.finalQ);
    }
    return null;
  }, [result, iterIndex]);

  const currentIter = useMemo(() => {
    if (!result || result.type === 'single-eval') return null;
    return result.iterations[iterIndex] ?? null;
  }, [result, iterIndex]);

  return (
    <InteractiveDemo title="MC Basic：策略评估与策略迭代">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            {displayPolicy ? (
              <GridWorld config={config} policy={displayPolicy} values={displayValues ?? undefined} showValues className="max-w-full" />
            ) : (
              <div className="text-gray-500">点击运行，查看 MC Basic 的策略与值函数</div>
            )}
          </div>
          {result?.type === 'policy-iteration' && (
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-800">
                  策略迭代过程 —— 第 {iterIndex + 1}/{result.iterations.length} 轮
                </h3>
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={iterIndex === 0}
                    onClick={() => setIterIndex((v) => v - 1)}
                  >
                    上一步
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={iterIndex >= result.iterations.length - 1}
                    onClick={() => setIterIndex((v) => v + 1)}
                  >
                    下一步
                  </Button>
                </div>
              </div>
              <div className="mb-2">
                <input
                  type="range"
                  min={0}
                  max={result.iterations.length - 1}
                  value={iterIndex}
                  onChange={(e) => setIterIndex(Number(e.target.value))}
                  className="w-full"
                />
              </div>
              {currentIter && (
                <div className="text-xs text-gray-600 space-y-1">
                  <div>
                    <strong>π_{currentIter.iteration}（policyBefore）→ </strong>
                    MC 策略评估 → <strong>q_{'{π_' + currentIter.iteration + '}'}</strong> → 贪心改进 → <strong>π_{currentIter.iteration + 1}（policyAfter）</strong>
                  </div>
                  <div className="mt-2">
                    策略稳定性：{currentIter.policyStable ? <span className="text-green-600 font-semibold">已稳定 ✓</span> : <span className="text-amber-600">未稳定</span>}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm text-gray-700 block mb-1">算法模式</label>
                <Select value={mode} onValueChange={(v) => setMode(v as 'single-eval' | 'policy-iteration')}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="policy-iteration">策略迭代（完整算法 5.1）</SelectItem>
                    <SelectItem value="single-eval">单步策略评估</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-gray-500 mt-1">
                  {mode === 'policy-iteration'
                    ? '反复执行"评估 + 贪心改进"直到策略收敛。'
                    : '仅对当前（随机）策略做一次 MC 评估，不做改进。'}
                </p>
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1">
                  <span>每对 (s,a) 采样回合数</span>
                  <span className="font-mono">{episodesPerPair}</span>
                </div>
                <Slider value={[episodesPerPair]} min={1} max={100} step={1} onValueChange={([v]) => setEpisodesPerPair(v)} />
              </div>
              <Button size="sm" onClick={run} className="w-full bg-blue-600 hover:bg-blue-700">
                运行 MC Basic
              </Button>
            </CardContent>
          </Card>
          {result && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">统计</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-gray-700 space-y-1">
                {result.type === 'single-eval' ? (
                  <>
                    <div>总样本数：{(config.rows * config.cols * 5 * episodesPerPair).toLocaleString()}</div>
                    <div>RMSE(q_est, q_π)：<span className="font-mono">{result.rmse.toFixed(4)}</span></div>
                    <div className="text-xs text-gray-500 mt-1">
                      真实值由模型计算，仅用于验证 MC 估计；MC 算法训练本身不使用模型。
                    </div>
                  </>
                ) : (
                  <>
                    <div>策略迭代轮数：<span className="font-mono font-semibold">{result.iterations.length}</span></div>
                    <div>策略是否稳定：<span className="font-mono">{result.stable ? '是 ✓' : '否'}</span></div>
                    <div>最终 RMSE(q_est, q_π)：<span className="font-mono">{result.rmse.toFixed(4)}</span></div>
                    <div className="text-xs text-gray-500 mt-1">
                      真实值由模型计算，仅用于验证 MC 估计；MC 算法训练本身不使用模型。
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- MC Exploring Starts -------------------
function MCExploringStartsDemo() {
  const config = DEFAULT_CONFIG;
  const qStar = useMemo(() => estimateTrueActionValues(config), [config]);
  const [visitMode, setVisitMode] = useState<VisitMode>('first-visit');
  const [learnerState, setLearnerState] = useState<MCLearnerState>(() => createMCLearnerState(config));
  const [rmseHistory, setRmseHistory] = useState<LearningPoint[]>(() => {
    const init = createMCLearnerState(config);
    return [{ episode: 0, rmse: qTableRMSE(init.q, qStar) }];
  });
  const [lastTrajectory, setLastTrajectory] = useState<{ state: number; action: Action; reward: number; nextState: number }[]>([]);
  const learnerRef = useRef(learnerState);
  learnerRef.current = learnerState;

  const policy = useMemo(() => greedyPolicy(learnerState.q), [learnerState]);
  const stateValues = useMemo(() => actionValueToStateValue(learnerState.q), [learnerState]);

  const runEpisodes = useCallback(
    (n: number) => {
      const newState = runMCExploringStartsEpisodes(
        learnerRef.current,
        config,
        n,
        30,
        visitMode
      );
      const rmse = qTableRMSE(newState.q, qStar);
      setLearnerState(newState);
      setRmseHistory((prev) => [...prev, { episode: newState.episodesCompleted, rmse }]);
      // Generate a sample trajectory for display
      const traj = mcExploringStarts(config, 1, 30, visitMode);
      setLastTrajectory(traj.lastTrajectory);
    },
    [config, qStar, visitMode]
  );

  function reset() {
    const init = createMCLearnerState(config);
    setLearnerState(init);
    setRmseHistory([{ episode: 0, rmse: qTableRMSE(init.q, qStar) }]);
    setLastTrajectory([]);
  }

  return (
    <MCDemoShell
      title="MC Exploring Starts：随机初始状态-动作（增量训练）"
      policy={policy}
      values={stateValues}
      episodeCount={learnerState.episodesCompleted}
      visitMode={visitMode}
      onVisitModeChange={setVisitMode}
      onRun={runEpisodes}
      onReset={reset}
      chartData={rmseHistory}
      lastTrajectory={lastTrajectory}
    />
  );
}

// ------------------- MC ε-Greedy -------------------
function MCEpsilonGreedyDemo() {
  const config = DEFAULT_CONFIG;
  const qStar = useMemo(() => estimateTrueActionValues(config), [config]);
  const [epsilon, setEpsilon] = useState(0.3);
  const [schedule, setSchedule] = useState<EpsilonSchedule>('fixed');
  const [visitMode, setVisitMode] = useState<VisitMode>('first-visit');
  const [learnerState, setLearnerState] = useState<MCLearnerState>(() => createMCLearnerState(config));
  const [rmseHistory, setRmseHistory] = useState<LearningPoint[]>(() => {
    const init = createMCLearnerState(config);
    return [{ episode: 0, rmse: qTableRMSE(init.q, qStar) }];
  });
  const [lastTrajectory, setLastTrajectory] = useState<{ state: number; action: Action; reward: number; nextState: number }[]>([]);
  const learnerRef = useRef(learnerState);
  learnerRef.current = learnerState;

  const policy = useMemo(() => epsilonGreedyPolicy(learnerState.q, learnerState.currentEpsilon), [learnerState]);
  const stateValues = useMemo(() => actionValueToStateValue(learnerState.q), [learnerState]);

  const runEpisodes = useCallback(
    (n: number) => {
      const newState = runMCEpsilonGreedyEpisodes(
        learnerRef.current,
        config,
        n,
        30,
        schedule,
        epsilon,
        visitMode
      );
      const rmse = qTableRMSE(newState.q, qStar);
      setLearnerState(newState);
      setRmseHistory((prev) => [...prev, { episode: newState.episodesCompleted, rmse }]);
      // Generate a sample trajectory for display
      const traj = mcEpsilonGreedy(config, 1, 30, newState.currentEpsilon, visitMode, schedule);
      setLastTrajectory(traj.lastTrajectory);
    },
    [config, epsilon, schedule, qStar, visitMode]
  );

  function reset() {
    const init = createMCLearnerState(config);
    setLearnerState(init);
    setRmseHistory([{ episode: 0, rmse: qTableRMSE(init.q, qStar) }]);
    setLastTrajectory([]);
  }

  return (
    <MCDemoShell
      title="MC ε-Greedy：持续探索的模型-free 控制（增量训练）"
      policy={policy}
      values={stateValues}
      episodeCount={learnerState.episodesCompleted}
      visitMode={visitMode}
      onVisitModeChange={setVisitMode}
      onRun={runEpisodes}
      onReset={reset}
      chartData={rmseHistory}
      lastTrajectory={lastTrajectory}
      epsilon={epsilon}
      onEpsilonChange={setEpsilon}
      schedule={schedule}
      onScheduleChange={setSchedule}
      currentEpsilon={learnerState.currentEpsilon}
    />
  );
}

// ------------------- Shared MC demo shell -------------------
interface MCDemoShellProps {
  title: string;
  policy: number[][];
  values: number[];
  episodeCount: number;
  visitMode: VisitMode;
  onVisitModeChange: (mode: VisitMode) => void;
  onRun: (n: number) => void;
  onReset: () => void;
  chartData: { episode: number; rmse: number }[];
  lastTrajectory: { state: number; action: Action; reward: number; nextState: number }[];
  epsilon?: number;
  onEpsilonChange?: (v: number) => void;
  schedule?: EpsilonSchedule;
  onScheduleChange?: (s: EpsilonSchedule) => void;
  currentEpsilon?: number;
}

function MCDemoShell(props: MCDemoShellProps) {
  const {
    title,
    policy,
    values,
    episodeCount,
    visitMode,
    onVisitModeChange,
    onRun,
    onReset,
    chartData,
    lastTrajectory,
    epsilon,
    onEpsilonChange,
    schedule,
    onScheduleChange,
    currentEpsilon,
  } = props;

  return (
    <InteractiveDemo title={title}>
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld config={DEFAULT_CONFIG} policy={policy} values={values} showValues className="max-w-full" />
          </div>
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <LineChart
              data={chartData}
              xKey="episode"
              xLabel="回合数"
              yLabel="RMSE(q, q*)"
              series={[{ key: 'rmse', name: 'RMSE', color: '#2563eb' }]}
              height={200}
            />
          </div>
          {lastTrajectory.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <h3 className="text-sm font-semibold text-gray-800 mb-2">最近一条轨迹</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="text-gray-500 border-b">
                    <tr>
                      <th className="py-1">步</th>
                      <th>状态</th>
                      <th>动作</th>
                      <th>奖励</th>
                    </tr>
                  </thead>
                  <tbody className="text-gray-700">
                    {lastTrajectory.slice(0, 12).map((step, idx) => (
                      <tr key={idx} className="border-b last:border-0">
                        <td className="py-1">{idx}</td>
                        <td>s{step.state + 1}</td>
                        <td>{ACTION_NAMES[step.action]}</td>
                        <td>{step.reward.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">统计信息</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-gray-700">
              <div>已采样回合：<span className="font-mono font-semibold">{episodeCount}</span></div>
              <div>折扣因子：<span className="font-mono">{DEFAULT_CONFIG.gamma}</span></div>
              {schedule && (
                <div>当前 ε 调度：<span className="font-mono">{schedule === 'fixed' ? '固定' : schedule === 'decaying-with-floor' ? '衰减（带下限）' : 'GLIE'}</span></div>
              )}
              {currentEpsilon !== undefined && (
                <div>当前 ε 值：<span className="font-mono">{currentEpsilon.toFixed(4)}</span></div>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">更新方式</CardTitle>
            </CardHeader>
            <CardContent>
              <Select
                value={visitMode}
                onValueChange={(v) => onVisitModeChange(v as VisitMode)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="选择 visit 模式" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="first-visit">First-visit</SelectItem>
                  <SelectItem value="every-visit">Every-visit</SelectItem>
                </SelectContent>
              </Select>
            </CardContent>
          </Card>
          {onEpsilonChange !== undefined && epsilon !== undefined && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">探索参数 <KaTeX math={String.raw`\varepsilon`} /></CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Slider value={[epsilon]} min={0} max={1} step={0.05} onValueChange={([v]) => onEpsilonChange(v)} />
                <div className="text-center font-mono text-sm text-gray-700">ε₀ = {epsilon.toFixed(2)}</div>
                {onScheduleChange && schedule !== undefined && (
                  <div>
                    <label className="text-xs text-gray-600 block mb-1">ε 调度方式</label>
                    <Select value={schedule} onValueChange={(v) => onScheduleChange(v as EpsilonSchedule)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="fixed">固定 ε</SelectItem>
                        <SelectItem value="decaying-with-floor">带最小探索率的衰减（不是 GLIE）</SelectItem>
                        <SelectItem value="glie">GLIE（ε → 0）</SelectItem>
                      </SelectContent>
                    </Select>
                    {schedule === 'fixed' && (
                      <p className="text-xs text-gray-500 mt-1">ε_k = ε₀，探索率始终不变。</p>
                    )}
                    {schedule === 'decaying-with-floor' && (
                      <p className="text-xs text-gray-500 mt-1">
                        ε_k = max(ε_min, ε₀ / √(k+1))，带最小探索率的衰减，不是 GLIE。
                      </p>
                    )}
                    {schedule === 'glie' && (
                      <p className="text-xs text-gray-500 mt-1">
                        ε_k = ε₀ / √(k+1)，ε_k → 0，严格 GLIE。
                      </p>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          )}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">采样控制</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button size="sm" onClick={() => onRun(10)} className="w-full">运行 10 个回合</Button>
              <Button size="sm" variant="outline" onClick={() => onRun(50)} className="w-full">运行 50 个回合</Button>
              <Button size="sm" variant="outline" onClick={() => onRun(100)} className="w-full">运行 100 个回合</Button>
              <Button size="sm" variant="outline" onClick={onReset} className="w-full">重置</Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}
