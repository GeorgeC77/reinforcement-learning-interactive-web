import { useState, useCallback, useMemo, useRef } from 'react';
import { Dices, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import LineChart from '@/components/LineChart';
import {
  DEFAULT_CONFIG, ACTION_NAMES,
  type Policy,
  epsilonGreedyPolicy, greedyPolicy, policyWeightedStateValues,
  mcBasic, type MCBasicIteration, mcBasicPolicyIteration,
  createMCLearnerState, runMCExploringStartsEpisodes, runMCEpsilonGreedyEpisodes,
  type MCLearnerState, type EpsilonSchedule, type TrajectoryStep,
  estimateTrueActionValues, solveStateValues, computeQValues, qTableRMSE,
} from '@/lib/rl/gridworld';

type TabKey = 'basic' | 'exploring' | 'epsilon';
type VisitMode = 'first-visit' | 'every-visit';
interface LearningPoint { episode: number; rmse: number; }
const HORIZON_OPTIONS = [10, 20, 30, 50, 100, 200];

export default function Chapter05MonteCarloPage() {
  const [activeTab, setActiveTab] = useState<TabKey>('basic');
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4"><div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center"><Dices className="w-8 h-8 text-blue-600" /></div></div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">第 5 章 蒙特卡洛方法</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">蒙特卡洛方法不依赖环境模型，通过采样轨迹并用样本回报估计状态价值或动作价值。</p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2"><ShieldAlert className="w-4 h-4" />本内容仅供教学与非商业学习使用。</p>
      </section>
      <section className="space-y-4">
        <FormulaCard title="蒙特卡洛回报" formula={<KaTeX math={String.raw`G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots + \gamma^{T-t-1} r_T`} display />} description="从时刻 t 开始，把折扣后的未来奖励相加得到一个样本回报。" />
        <FormulaCard title="样本平均更新" formula={<KaTeX math={String.raw`q(s,a) \leftarrow q(s,a) + \frac{1}{N(s,a)} \bigl(G - q(s,a)\bigr)`} display />} description="每遇到一次 (s,a)，用新样本逐步修正估计。" />
        <FormulaCard title="ε-贪心策略" formula={<KaTeX math={String.raw`\pi(a|s) = \begin{cases} 1-\varepsilon + \frac{\varepsilon}{|\mathcal{A}|}, & a = \arg\max_{a'} q(s,a') \\ \frac{\varepsilon}{|\mathcal{A}|}, & \text{否则} \end{cases}`} display />} description="大部分时候选择当前最优动作，偶尔随机探索。" />
      </section>
      <Tabs value={activeTab} onValueChange={(v)=>setActiveTab(v as TabKey)}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="basic">MC Basic</TabsTrigger>
          <TabsTrigger value="exploring">Exploring Starts</TabsTrigger>
          <TabsTrigger value="epsilon">ε-Greedy</TabsTrigger>
        </TabsList>
        <TabsContent value="basic" className="mt-4"><MCBasicDemo /></TabsContent>
        <TabsContent value="exploring" className="mt-4"><MCExploringStartsDemo /></TabsContent>
        <TabsContent value="epsilon" className="mt-4"><MCEpsilonGreedyDemo /></TabsContent>
      </Tabs>
      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">观察与思考</h2>
        <ul className="space-y-3 text-gray-700">
          <li><strong>MC Basic：</strong>完整的策略迭代算法。每轮对每对 (s,a) 采样回合估计 q_π，然后贪心改进，直到策略稳定。</li>
          <li><strong>Exploring Starts：</strong>用随机初始 (s,a) 保证每对动作值都能被访问，随后遵循当前贪心策略。</li>
          <li><strong>ε-Greedy：</strong>当无法保证 exploring starts 时，用 ε-贪心持续探索。固定 ε&gt;0 时不保证收敛到 q*，仅收敛到最佳 ε-soft 策略。</li>
          <li><strong>First-visit vs Every-visit：</strong>First-visit 只使用每个回合中首次访问 (s,a) 后的回报；Every-visit 使用该回合中的全部访问，因此能利用更多样本，但同一回合内这些回报通常相关。两种方法在适当条件下均可收敛，其有限样本偏差与方差取决于具体采样过程。</li>
        </ul>
      </section>
    </div>
  );
}

// ------------------- MC Basic -------------------
function MCBasicDemo() {
  const config = DEFAULT_CONFIG;
  const [episodesPerPair, setEpisodesPerPair] = useState(20);
  const [horizonT, setHorizonT] = useState(30);
  const [mode, setMode] = useState<'single-eval'|'policy-iteration'>('policy-iteration');
  const [result, setResult] = useState<{type:'single-eval';q:number[][];counts:number[][];rmse:number;policy:Policy}|{type:'policy-iteration';iterations:MCBasicIteration[];finalPolicy:Policy;finalQ:number[][];finalQPolicy:Policy;rmse:number;stable:boolean}|null>(null);
  const [iterIndex, setIterIndex] = useState(0);
  const [showImproved, setShowImproved] = useState(false);

  function run() {
    if (mode === 'single-eval') {
      const { policy, qValues, returns } = mcBasic(config, episodesPerPair, horizonT);
      const counts = returns.map(s=>s.map(a=>a.length));
      const vTrue = solveStateValues(policy, config); const qTrue = computeQValues(vTrue, config);
      setResult({ type:'single-eval', q:qValues, counts, rmse:qTableRMSE(qValues,qTrue), policy });
      setShowImproved(false); setIterIndex(0);
    } else {
      const { iterations, finalPolicy, finalQ, finalQPolicy } = mcBasicPolicyIteration(config, episodesPerPair, horizonT, 20);
      const vTrue = solveStateValues(finalQPolicy, config); const qTrue = computeQValues(vTrue, config);
      setResult({ type:'policy-iteration', iterations, finalPolicy, finalQ, finalQPolicy, rmse:qTableRMSE(finalQ,qTrue), stable:iterations[iterations.length-1]?.policyStable??false });
      setIterIndex(0);
    }
  }

  // Top grid: always π_k + V_{π_k} in policy-iteration mode
  const displayPolicy = useMemo(() => {
    if(!result) return null;
    if(result.type==='single-eval') return showImproved ? greedyPolicy(result.q) : result.policy;
    const ci = result.iterations[iterIndex];
    return ci?.policyBefore ?? result.finalQPolicy;
  },[result,iterIndex,showImproved]);

  const displayValues = useMemo(() => {
    if(!result) return null;
    if(result.type==='single-eval') {
      const pol = showImproved ? greedyPolicy(result.q) : result.policy;
      return policyWeightedStateValues(result.q, pol);
    }
    const ci = result.iterations[iterIndex];
    const qEst = ci?.qEstimate ?? result.finalQ;
    const pol = ci?.policyBefore ?? result.finalQPolicy;
    return policyWeightedStateValues(qEst, pol);
  },[result,iterIndex,showImproved]);

  const currentIter = result?.type==='policy-iteration' ? result.iterations[iterIndex] ?? null : null;
  const stableButMaxed = result?.type==='policy-iteration' && !result.stable && result.iterations.length>=20;

  return (<InteractiveDemo title="MC Basic：策略评估与策略迭代">
    <p className="text-xs text-gray-500 mb-2">本示例为 continuing task。使用长度为 T 的有限截断轨迹近似无限期折扣回报。{mode==='single-eval'&&' RMSE 同时包含采样误差和有限轨迹截断误差。'}</p>
    <div className="grid lg:grid-cols-[1fr_340px] gap-6">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
          {displayPolicy ? (<>
            <GridWorld config={config} policy={displayPolicy} values={displayValues??undefined} showValues className="max-w-full" />
            {result?.type==='single-eval' && <div className="mt-2 text-xs text-gray-500">{showImproved?'贪心改进策略':'被评估策略 π（随机策略）'}</div>}
            {result?.type==='policy-iteration' && <div className="mt-2 text-xs text-gray-500">π<sub>{currentIter?.iteration}</sub> 与 V<sub>π<sub>{currentIter?.iteration}</sub></sub></div>}
          </>) : <div className="text-gray-500">点击运行，查看 MC Basic 的策略与值函数</div>}
        </div>
        {result?.type==='single-eval' && <div className="flex items-center gap-2 text-xs"><label className="flex items-center gap-1 cursor-pointer"><input type="checkbox" checked={showImproved} onChange={()=>setShowImproved(!showImproved)} />显示贪心改进策略（根据 q_π 的一次改进 π'）</label></div>}
        {currentIter && (<div className="bg-white rounded-xl border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-800">策略迭代 —— 第 {iterIndex+1}/{result?.type==='policy-iteration'?result.iterations.length:0} 轮</h3>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline" disabled={iterIndex===0} onClick={()=>setIterIndex(v=>v-1)}>上一步</Button>
              <Button size="sm" variant="outline" disabled={iterIndex>=(result?.type==='policy-iteration'?result.iterations.length:1)-1} onClick={()=>setIterIndex(v=>v+1)}>下一步</Button>
            </div>
          </div>
          <input type="range" min={0} max={(result?.type==='policy-iteration'?result.iterations.length:1)-1} value={iterIndex} onChange={e=>setIterIndex(Number(e.target.value))} className="w-full mb-3" />
          <div className="text-xs text-gray-600 mb-3">π<sub>{currentIter.iteration}</sub> → MC 策略评估 → q<sub>π<sub>{currentIter.iteration}</sub></sub> → 贪心改进 → π<sub>{currentIter.iteration+1}</sub></div>
          <div className="grid grid-cols-3 gap-3">
            <div className="text-center"><div className="text-[10px] font-medium text-gray-500 mb-1">π<sub>{currentIter.iteration}</sub> (评估前)</div><GridWorld config={config} policy={currentIter.policyBefore} className="max-w-full" /></div>
            <div className="text-center"><div className="text-[10px] font-medium text-gray-500 mb-1">V<sub>π<sub>{currentIter.iteration}</sub></sub></div><GridWorld config={config} policy={currentIter.policyBefore} values={policyWeightedStateValues(currentIter.qEstimate,currentIter.policyBefore)} showValues className="max-w-full" /></div>
            <div className="text-center"><div className="text-[10px] font-medium text-gray-500 mb-1">π<sub>{currentIter.iteration+1}</sub> (改进后)</div><GridWorld config={config} policy={currentIter.policyAfter} className="max-w-full" /></div>
          </div>
          <div className="mt-3 text-xs text-gray-600 space-y-1">
            <div>本轮有 <strong>{currentIter.changedStateCount}</strong> 个状态的动作发生变化</div>
            {currentIter.policyStable ? <div className="text-green-600 font-semibold">策略已稳定 ✓</div> : <div className="text-amber-600">策略尚未稳定</div>}
          </div>
        </div>)}
      </div>
      <div className="space-y-4">
        <Card><CardHeader className="pb-2"><CardTitle className="text-base">设置</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div><label className="text-sm text-gray-700 block mb-1">算法模式</label>
              <Select value={mode} onValueChange={v=>setMode(v as 'single-eval'|'policy-iteration')}><SelectTrigger><SelectValue/></SelectTrigger><SelectContent><SelectItem value="policy-iteration">策略迭代（完整算法 5.1）</SelectItem><SelectItem value="single-eval">单步策略评估</SelectItem></SelectContent></Select></div>
            <div><div className="flex justify-between text-sm text-gray-700 mb-1"><span>每对 (s,a) 采样回合数</span><span className="font-mono">{episodesPerPair}</span></div><Slider value={[episodesPerPair]} min={1} max={100} step={1} onValueChange={([v])=>setEpisodesPerPair(v)}/></div>
            <div><label className="text-sm text-gray-700 block mb-1">轨迹长度 T</label><Select value={String(horizonT)} onValueChange={v=>setHorizonT(Number(v))}><SelectTrigger><SelectValue/></SelectTrigger><SelectContent>{HORIZON_OPTIONS.map(t=><SelectItem key={t} value={String(t)}>T={t}</SelectItem>)}</SelectContent></Select></div>
            <Button size="sm" onClick={run} className="w-full bg-blue-600 hover:bg-blue-700">运行 MC Basic</Button>
          </CardContent></Card>
        {result && <Card><CardHeader className="pb-2"><CardTitle className="text-base">统计</CardTitle></CardHeader>
          <CardContent className="text-sm text-gray-700 space-y-1">
            {result.type==='single-eval' ? (<><div>总样本数：{(config.rows*config.cols*5*episodesPerPair).toLocaleString()}</div><div>RMSE(q_est, q_π)：<span className="font-mono">{result.rmse.toFixed(4)}</span></div><div className="text-xs text-gray-500 mt-1">真实值由模型计算仅用于验证；MC 训练本身不使用模型。RMSE 同时包含采样误差与有限轨迹截断误差。</div></>)
            : (<><div>策略迭代轮数：<span className="font-mono font-semibold">{result.iterations.length}</span></div><div>策略是否稳定：<span className="font-mono">{result.stable?'是 ✓':(stableButMaxed?'尚未观察到稳定':'否')}</span></div><div>最终 RMSE(q_est, q_π)：<span className="font-mono">{result.rmse.toFixed(4)}</span></div><div className="text-xs text-gray-500 mt-1">真实值由模型计算仅用于验证。</div>{stableButMaxed&&<div className="text-xs text-amber-600 mt-1">在当前有限样本和最大迭代轮数下，尚未观察到策略稳定。</div>}</>)}
          </CardContent></Card>}
      </div>
    </div>
  </InteractiveDemo>);
}

// ------------------- MC Exploring Starts -------------------
function MCExploringStartsDemo() {
  const config = DEFAULT_CONFIG;
  const qStar = useMemo(()=>estimateTrueActionValues(config),[config]);
  const [visitMode, setVisitMode] = useState<VisitMode>('first-visit');
  const [horizonT, setHorizonT] = useState(30);
  const [learnerState, setLearnerState] = useState<MCLearnerState>(()=>createMCLearnerState(config));
  const [rmseHistory, setRmseHistory] = useState<LearningPoint[]>(()=>[{episode:0,rmse:qTableRMSE(createMCLearnerState(config).q,qStar)}]);
  const learnerRef = useRef(learnerState); learnerRef.current = learnerState;

  const policy = useMemo(()=>learnerState.policy,[learnerState]);
  const stateValues = useMemo(()=>policyWeightedStateValues(learnerState.q,policy),[learnerState,policy]);

  const runEpisodes = useCallback((n:number)=>{
    const ns = runMCExploringStartsEpisodes(learnerRef.current,config,n,horizonT,visitMode);
    setLearnerState(ns);
    setRmseHistory(prev=>[...prev,{episode:ns.episodesCompleted,rmse:qTableRMSE(ns.q,qStar)}]);
  },[config,qStar,visitMode,horizonT]);

  function reset() {
    const init = createMCLearnerState(config);
    setLearnerState(init); setRmseHistory([{episode:0,rmse:qTableRMSE(init.q,qStar)}]);
  }

  return (<MCDemoShell title="MC Exploring Starts：随机初始状态-动作（增量训练）"
    policy={policy} values={stateValues} episodeCount={learnerState.episodesCompleted}
    visitMode={visitMode} onVisitModeChange={setVisitMode} onRun={runEpisodes} onReset={reset}
    chartData={rmseHistory} lastTrajectory={learnerState.lastTrajectory}
    horizonT={horizonT} onHorizonTChange={setHorizonT}/>);
}

// ------------------- MC ε-Greedy -------------------
function MCEpsilonGreedyDemo() {
  const config = DEFAULT_CONFIG;
  const qStar = useMemo(()=>estimateTrueActionValues(config),[config]);
  const [epsilon, setEpsilon] = useState(0.3);
  const [schedule, setSchedule] = useState<EpsilonSchedule>('fixed');
  const [visitMode, setVisitMode] = useState<VisitMode>('first-visit');
  const [horizonT, setHorizonT] = useState(30);
  const [learnerState, setLearnerState] = useState<MCLearnerState>(()=>createMCLearnerState(config, epsilon));
  const [rmseHistory, setRmseHistory] = useState<LearningPoint[]>(()=>[{episode:0,rmse:qTableRMSE(createMCLearnerState(config,epsilon).q,qStar)}]);
  const learnerRef = useRef(learnerState); learnerRef.current = learnerState;

  const effectiveEpsilon = learnerState.episodesCompleted===0 ? epsilon : learnerState.currentEpsilon;
  const policy = useMemo(()=>epsilonGreedyPolicy(learnerState.q, effectiveEpsilon),[learnerState, effectiveEpsilon]);
  const stateValues = useMemo(()=>policyWeightedStateValues(learnerState.q, policy),[learnerState, policy]);

  const runEpisodes = useCallback((n:number)=>{
    const ns = runMCEpsilonGreedyEpisodes(learnerRef.current,config,n,horizonT,schedule,epsilon,visitMode);
    setLearnerState(ns);
    setRmseHistory(prev=>[...prev,{episode:ns.episodesCompleted,rmse:qTableRMSE(ns.q,qStar)}]);
  },[config,epsilon,schedule,qStar,visitMode,horizonT]);

  function reset() {
    const init = createMCLearnerState(config, epsilon);
    setLearnerState(init); setRmseHistory([{episode:0,rmse:qTableRMSE(init.q,qStar)}]);
  }

  return (<MCDemoShell title="MC ε-Greedy：持续探索的模型-free 控制（增量训练）"
    policy={policy} values={stateValues} episodeCount={learnerState.episodesCompleted}
    visitMode={visitMode} onVisitModeChange={setVisitMode} onRun={runEpisodes} onReset={reset}
    chartData={rmseHistory} lastTrajectory={learnerState.lastTrajectory}
    epsilon={epsilon} onEpsilonChange={setEpsilon} schedule={schedule} onScheduleChange={setSchedule}
    currentEpsilon={effectiveEpsilon} horizonT={horizonT} onHorizonTChange={setHorizonT}
    rmseNote={schedule==='fixed'?'固定 ε>0 时，学习目标是最佳 ε-soft 策略，Q 与 q* 的 RMSE 不一定趋于 0。':undefined}/>);
}

// ------------------- Shared MC demo shell -------------------
interface MCDemoShellProps {
  title: string; policy: number[][]; values: number[]; episodeCount: number;
  visitMode: VisitMode; onVisitModeChange: (mode: VisitMode) => void;
  onRun: (n: number) => void; onReset: () => void;
  chartData: { episode: number; rmse: number }[];
  lastTrajectory: TrajectoryStep[];
  epsilon?: number; onEpsilonChange?: (v: number) => void;
  schedule?: EpsilonSchedule; onScheduleChange?: (s: EpsilonSchedule) => void;
  currentEpsilon?: number; horizonT?: number; onHorizonTChange?: (v: number) => void;
  rmseNote?: string;
}

function MCDemoShell(props: MCDemoShellProps) {
  const { title,policy,values,episodeCount,visitMode,onVisitModeChange,onRun,onReset,chartData,lastTrajectory,
    epsilon,onEpsilonChange,schedule,onScheduleChange,currentEpsilon,horizonT,onHorizonTChange,rmseNote } = props;
  return (<InteractiveDemo title={title}>
    <p className="text-xs text-gray-500 mb-2">本示例为 continuing task。使用长度为 T 的有限截断轨迹近似无限期折扣回报。在其他条件相同时，增大 T 通常会减小截断误差，但会增加计算成本并可能提高样本回报的方差。</p>
    <div className="grid lg:grid-cols-[1fr_340px] gap-6">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
          <GridWorld config={DEFAULT_CONFIG} policy={policy} values={values} showValues className="max-w-full" />
        </div>
        <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
          <LineChart data={chartData} xKey="episode" xLabel="回合数" yLabel="RMSE(q, q*)" series={[{ key:'rmse', name:'RMSE', color:'#2563eb' }]} height={200} />
          {rmseNote && <p className="text-[10px] text-amber-600 mt-1">{rmseNote}</p>}
        </div>
        {lastTrajectory.length>0 && (
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <h3 className="text-sm font-semibold text-gray-800 mb-2">最近一条训练轨迹</h3>
            <div className="overflow-x-auto"><table className="w-full text-xs text-left">
              <thead className="text-gray-500 border-b"><tr><th className="py-1">步</th><th>状态</th><th>动作</th><th>奖励</th></tr></thead>
              <tbody className="text-gray-700">{lastTrajectory.slice(0,12).map((step,idx)=>(<tr key={idx} className="border-b last:border-0"><td className="py-1">{idx}</td><td>s{step.state+1}</td><td>{ACTION_NAMES[step.action]}</td><td>{step.reward.toFixed(2)}</td></tr>))}</tbody>
            </table></div>
          </div>)}
      </div>
      <div className="space-y-4">
        <Card><CardHeader className="pb-2"><CardTitle className="text-base">统计信息</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-sm text-gray-700">
            <div>已采样回合：<span className="font-mono font-semibold">{episodeCount}</span></div>
            <div>折扣因子：<span className="font-mono">{DEFAULT_CONFIG.gamma}</span></div>
            {horizonT!==undefined&&<div>轨迹长度 T：<span className="font-mono">{horizonT}</span></div>}
            {schedule&&<div>ε 调度：<span className="font-mono">{schedule==='fixed'?'固定':schedule==='decaying-with-floor'?'衰减（带下限）':'GLIE'}</span></div>}
            {currentEpsilon!==undefined&&<div>当前 ε：<span className="font-mono">{currentEpsilon.toFixed(4)}</span></div>}
          </CardContent></Card>
        <Card><CardHeader className="pb-2"><CardTitle className="text-base">更新方式</CardTitle></CardHeader>
          <CardContent><Select value={visitMode} onValueChange={v=>onVisitModeChange(v as VisitMode)}><SelectTrigger><SelectValue placeholder="选择 visit 模式"/></SelectTrigger><SelectContent><SelectItem value="first-visit">First-visit</SelectItem><SelectItem value="every-visit">Every-visit</SelectItem></SelectContent></Select></CardContent></Card>
        {onEpsilonChange!==undefined&&epsilon!==undefined&&(<Card><CardHeader className="pb-2"><CardTitle className="text-base">探索参数 <KaTeX math={String.raw`\varepsilon`}/></CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <Slider value={[epsilon]} min={0} max={1} step={0.05} onValueChange={([v])=>onEpsilonChange(v)}/>
            <div className="text-center font-mono text-sm text-gray-700">ε₀ = {epsilon.toFixed(2)}</div>
            {onScheduleChange&&schedule!==undefined&&(<div><label className="text-xs text-gray-600 block mb-1">ε 调度方式</label>
              <Select value={schedule} onValueChange={v=>onScheduleChange(v as EpsilonSchedule)}><SelectTrigger><SelectValue/></SelectTrigger>
                <SelectContent><SelectItem value="fixed">固定 ε</SelectItem><SelectItem value="decaying-with-floor">带最小探索率的衰减（不是 GLIE）</SelectItem><SelectItem value="glie">GLIE（ε → 0）</SelectItem></SelectContent></Select>
              {schedule==='fixed'&&<p className="text-xs text-gray-500 mt-1">ε_k = ε₀，探索率始终不变。</p>}
              {schedule==='decaying-with-floor'&&<p className="text-xs text-gray-500 mt-1">ε_k = max(ε_min, ε₀ / √(k+1))，不是 GLIE。</p>}
              {schedule==='glie'&&<p className="text-xs text-gray-500 mt-1">ε_k = ε₀ / √(k+1)，ε_k → 0，严格 GLIE。</p>}
            </div>)}
          </CardContent></Card>)}
        {onHorizonTChange!==undefined&&horizonT!==undefined&&(<Card><CardHeader className="pb-2"><CardTitle className="text-base">轨迹长度 T</CardTitle></CardHeader><CardContent><Select value={String(horizonT)} onValueChange={v=>onHorizonTChange(Number(v))}><SelectTrigger data-testid="mc-horizon-select"><SelectValue/></SelectTrigger><SelectContent>{HORIZON_OPTIONS.map(t=><SelectItem key={t} value={String(t)}>T={t}</SelectItem>)}</SelectContent></Select></CardContent></Card>)}
        <Card><CardHeader className="pb-2"><CardTitle className="text-base">采样控制</CardTitle></CardHeader>
          <CardContent className="space-y-2">
            <Button size="sm" onClick={()=>onRun(10)} className="w-full">运行 10 个回合</Button>
            <Button size="sm" variant="outline" onClick={()=>onRun(50)} className="w-full">运行 50 个回合</Button>
            <Button size="sm" variant="outline" onClick={()=>onRun(100)} className="w-full">运行 100 个回合</Button>
            <Button size="sm" variant="outline" onClick={onReset} className="w-full">重置</Button>
          </CardContent></Card>
      </div>
    </div>
  </InteractiveDemo>);
}
