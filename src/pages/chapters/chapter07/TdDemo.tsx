import { useEffect, useMemo, useState } from 'react';
import { Clock, ShieldAlert, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import KaTeX from '@/components/KaTeX';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import AlgorithmPlayer from '@/components/AlgorithmPlayer';
import LineChart from '@/components/LineChart';
import {
  DEFAULT_CONFIG,
  EPISODIC_PATH_CONFIG,
  ACTION_NAMES,
  deterministicPolicy,
  randomPolicy,
  epsilonGreedyPolicy,
  actionValueToStateValue,
  policyWeightedStateValues,
  policyBellmanResidualV,
  policyBellmanResidualQ,
  optimalBellmanResidualQ,
  epsilonAtEpisode,
  solveStateValues,
  estimateTrueActionValues,
  isTerminal,
  tdZeroPrediction,
  sarsa,
  qLearning,
  nStepSarsa,
  expectedSarsa,
  sarsaLambda,
  tdLambdaPrediction,
  type Action,
  type Policy,
  type GridWorldConfig,
  type EpsilonScheduleMode,

  type TDUpdateRecord,
  type PredictionFrame,
  type ControlFrame,
} from '@/lib/rl/gridworld';

export type AlgorithmKind =
  | 'td0'
  | 'sarsa'
  | 'nstep'
  | 'qlearning'
  | 'expected'
  | 'sarsa-lambda'
  | 'td-lambda';

export type AlgorithmCategory = 'main' | 'supplement' | 'extension';

export type AlgorithmDef = {
  key: AlgorithmKind;
  label: string;
  category: AlgorithmCategory;
};

type TdDemoProps = {
  title: string;
  subtitle: string;
  algorithms: AlgorithmDef[];
  defaultAlgorithm: AlgorithmKind;
};

const RIGHT_POLICY: Action[] = [1, 1, 1, 1, 1, 1, 1, 1, 1];
const GOAL_POLICY: Action[] = [1, 2, 2, 1, 2, 2, 1, 1, 4];
const H_OPTIONS = [10, 20, 30, 50, 100, 200];

function isPredictionAlgorithm(a: AlgorithmKind) {
  return a === 'td0' || a === 'td-lambda';
}

function getDefaultTask(a: AlgorithmKind): 'continuing' | 'episodic' {
  return isPredictionAlgorithm(a) ? 'continuing' : 'episodic';
}

function getDefaultValueMode(a: AlgorithmKind): 'behavior' | 'greedy' {
  return a === 'qlearning' ? 'greedy' : 'behavior';
}

function policyFromPreset(preset: string, numStates: number): Policy {
  if (preset === 'right') return deterministicPolicy(RIGHT_POLICY, 5);
  if (preset === 'random') return randomPolicy(numStates, 5);
  return deterministicPolicy(GOAL_POLICY, 5);

}

function computeGreedyAgreement(q: number[][], qStar: number[][], config: GridWorldConfig) {
  const tolerance = 1e-6;
  let total = 0;
  let match = 0;
  for (let s = 0; s < q.length; s++) {
    if (isTerminal(s, config)) continue;
    total++;
    const maxQ = Math.max(...q[s]);
    const greedyActions = q[s]
      .map((val, a) => ({ val, a }))
      .filter(({ val }) => Math.abs(val - maxQ) <= tolerance)
      .map(({ a }) => a);
    const maxQStar = Math.max(...qStar[s]);
    const optimalActions = qStar[s]
      .map((val, a) => ({ val, a }))
      .filter(({ val }) => Math.abs(val - maxQStar) <= tolerance)
      .map(({ a }) => a);
    const set = new Set(optimalActions);
    if (greedyActions.some((a) => set.has(a))) match++;
  }
  return total === 0 ? 0 : match / total;
}

function stateName(s: number) {
  return `s_{${s + 1}}`;
}

function actionName(a: number) {
  return ACTION_NAMES[a];
}

function formatVal(x: number) {
  return x.toFixed(3);
}

function UpdateFormula({
  update,
  algorithm,
  alpha,
  gamma,
}: {
  update: TDUpdateRecord;
  algorithm: AlgorithmKind;
  alpha: number;
  gamma: number;
}) {
  const s = stateName(update.state);
  const sp = stateName(update.nextState);
  const a = actionName(update.action);
  const old = formatVal(update.oldEstimate);
  const ne = formatVal(update.newEstimate);
  const r = formatVal(update.reward);
  const boot = formatVal(update.bootstrapValue);

  if (algorithm === 'td0' || algorithm === 'td-lambda') {
    return (
      <KaTeX
        math={String.raw`v(${s}) \leftarrow ${old} + ${alpha.toFixed(2)}\bigl[ ${r} + ${gamma.toFixed(2)}\cdot ${boot} - ${old} \bigr] = ${ne}`}
        display
      />
    );
  }

  if (algorithm === 'nstep') {
    const rewards = update.rewardTerms ?? [update.reward];
    const rewardTerms = rewards
      .map((rew, k) => `${k === 0 ? '' : String.raw`+ ${gamma.toFixed(2)}^{${k}} \cdot `}${formatVal(rew)}`)
      .join(' ');
    const bootStr =
      update.bootstrapState !== undefined && update.bootstrapAction !== undefined
        ? String.raw`Q(${stateName(update.bootstrapState)}, ${actionName(update.bootstrapAction)})`
        : boot;
    const bootTerm = update.bootstrapValue !== 0
      ? String.raw`+ ${gamma.toFixed(2)}^{${rewards.length}} \cdot ${bootStr}`
      : '';
    return (
      <KaTeX
        math={String.raw`Q(${s}, ${a}) \leftarrow ${old} + ${alpha.toFixed(2)}\bigl[ ${rewardTerms} ${bootTerm} - ${old} \bigr] = ${ne}`}
        display
      />
    );
  }

  if (algorithm === 'expected') {
    return (
      <KaTeX
        math={String.raw`Q(${s}, ${a}) \leftarrow ${old} + ${alpha.toFixed(2)}\bigl[ ${r} + ${gamma.toFixed(2)}\cdot ${boot} - ${old} \bigr] = ${ne}`}
        display
      />
    );
  }

  if (algorithm === 'sarsa' || algorithm === 'sarsa-lambda') {
    const nextA = update.nextAction !== undefined ? actionName(update.nextAction) : '?';
    const bootStr = update.done
      ? '0'
      : String.raw`Q(${sp}, ${nextA})`;
    return (
      <KaTeX
        math={String.raw`Q(${s}, ${a}) \leftarrow ${old} + ${alpha.toFixed(2)}\bigl[ ${r} + ${gamma.toFixed(2)}\cdot ${bootStr} - ${old} \bigr] = ${ne}`}
        display
      />
    );
  }

  // qlearning
  return (
    <KaTeX
      math={String.raw`Q(${s}, ${a}) \leftarrow ${old} + ${alpha.toFixed(2)}\bigl[ ${r} + ${gamma.toFixed(2)}\cdot ${boot} - ${old} \bigr] = ${ne}`}
      display
    />
  );
}

export default function TdDemo({ title, subtitle, algorithms, defaultAlgorithm }: TdDemoProps) {
  const [algorithm, setAlgorithm] = useState<AlgorithmKind>(defaultAlgorithm);
  const [alpha, setAlpha] = useState(0.2);
  const [epsilon, setEpsilon] = useState(0.3);
  const [epsilonMode, setEpsilonMode] = useState<EpsilonScheduleMode>('fixed');
  const [lambda, setLambda] = useState(0.8);
  const [nStep, setNStep] = useState(3);
  const [episodes, setEpisodes] = useState(100);
  const [horizonH, setHorizonH] = useState(30);
  const [seed, setSeed] = useState(1);
  const [task, setTask] = useState<'continuing' | 'episodic'>(getDefaultTask(defaultAlgorithm));
  const [policyPreset, setPolicyPreset] = useState<'goal' | 'random' | 'right'>('goal');
  const [viewMode, setViewMode] = useState<'transition' | 'episode'>('transition');
  const [valueMode, setValueMode] = useState<'behavior' | 'greedy'>(getDefaultValueMode(defaultAlgorithm));
  const [step, setStep] = useState(0);

  const isPrediction = isPredictionAlgorithm(algorithm);
  const config = task === 'continuing' ? DEFAULT_CONFIG : EPISODIC_PATH_CONFIG;
  const numStates = config.rows * config.cols;
  const policy = useMemo(() => policyFromPreset(policyPreset, numStates), [policyPreset, numStates]);

  useEffect(() => {
    setStep(0);
  }, [algorithm, alpha, epsilon, epsilonMode, lambda, nStep, episodes, horizonH, seed, task, policyPreset]);

  useEffect(() => {
    setTask(getDefaultTask(algorithm));
    setValueMode(getDefaultValueMode(algorithm));
    setPolicyPreset('goal');
  }, [algorithm]);

  const result = useMemo(() => {
    switch (algorithm) {
      case 'td0':
        return tdZeroPrediction(policy, config, alpha, horizonH, episodes, seed);
      case 'sarsa':
        return sarsa(config, alpha, epsilon, epsilonMode, horizonH, episodes, seed);
      case 'nstep':
        return nStepSarsa(config, alpha, epsilon, epsilonMode, nStep, horizonH, episodes, seed);
      case 'qlearning':
        return qLearning(config, alpha, epsilon, epsilonMode, horizonH, episodes, seed);
      case 'expected':
        return expectedSarsa(config, alpha, epsilon, epsilonMode, horizonH, episodes, seed);
      case 'sarsa-lambda':
        return sarsaLambda(config, alpha, epsilon, epsilonMode, lambda, horizonH, episodes, seed);
      case 'td-lambda':
        return tdLambdaPrediction(policy, config, alpha, lambda, horizonH, episodes, seed);
    }
  }, [algorithm, alpha, epsilon, epsilonMode, lambda, nStep, episodes, horizonH, seed, config, policy]);

  const totalSteps =
    viewMode === 'transition'
      ? Math.max(0, result.updates.length - 1)
      : Math.max(0, result.frames.length - 1);
  const safeStep = Math.min(step, totalSteps);
  const currentUpdate = viewMode === 'transition' ? result.updates[safeStep] : null;
  const currentFrame = viewMode === 'episode' ? result.frames[safeStep] : null;

  const display = useMemo(() => {
    if (currentUpdate) {
      if (isPrediction) {
        const values = currentUpdate.valuesAfter!;
        return {
          values,
          policy,
          highlightState: currentUpdate.state,
          highlightNextState: currentUpdate.nextState,
          highlightUpdatedState: currentUpdate.state,
          highlightAction: null as { state: number; action: number } | null,
          valueLabel: '状态价值',
        };
      }
      const q = currentUpdate.qAfter!;
      const behaviorPolicyBefore = currentUpdate.behaviorPolicyBefore ?? epsilonGreedyPolicy(q, epsilon);
      const behaviorPolicyAfter = currentUpdate.behaviorPolicyAfter ?? behaviorPolicyBefore;
      const values = valueMode === 'behavior'
        ? policyWeightedStateValues(q, behaviorPolicyAfter)
        : actionValueToStateValue(q);
      return {
        values,
        policy: behaviorPolicyBefore,
        highlightState: currentUpdate.state,
        highlightNextState: currentUpdate.nextState,
        highlightUpdatedState: currentUpdate.state,
        highlightAction: { state: currentUpdate.state, action: currentUpdate.action },
        valueLabel: valueMode === 'behavior' ? '行为策略价值（更新后）' : '贪心派生价值（更新后）',
      };
    }

    if (currentFrame) {
      if (currentFrame.kind === 'v') {
        return {
          values: currentFrame.values,
          policy: currentFrame.policy,
          highlightState: null,
          highlightNextState: null,
          highlightUpdatedState: null,
          highlightAction: null,
          valueLabel: '状态价值',
        };
      }
      const chosenPolicy = valueMode === 'behavior' ? currentFrame.behaviorPolicy : currentFrame.greedyPolicy;
      const values = valueMode === 'behavior'
        ? policyWeightedStateValues(currentFrame.qValues, chosenPolicy)
        : actionValueToStateValue(currentFrame.qValues);
      return {
        values,
        policy: chosenPolicy,
        highlightState: null,
        highlightNextState: null,
        highlightUpdatedState: null,
        highlightAction: null,
        valueLabel: valueMode === 'behavior' ? '行为策略价值' : '贪心派生价值',
      };
    }

    return {
      values: new Array(numStates).fill(0),
      policy: randomPolicy(numStates, 5),
      highlightState: null,
      highlightNextState: null,
      highlightUpdatedState: null,
      highlightAction: null,
      valueLabel: '',
    };
  }, [currentUpdate, currentFrame, isPrediction, policy, valueMode, epsilon, epsilonMode, numStates]);

  const transitionErrorData = useMemo(() => {
    return result.updates.map((u, i) => ({
      time: i,
      episode: u.episode,
      tdError: u.tdError,
    }));
  }, [result]);

  const residualData = useMemo(() => {
    return result.frames.slice(1).map((frame, i) => {
      let residual = 0;
      if (frame.kind === 'v') {
        residual = policyBellmanResidualV(frame.values, frame.policy, config);
      } else if (algorithm === 'qlearning') {
        residual = optimalBellmanResidualQ(frame.qValues, config);
      } else {
        residual = policyBellmanResidualQ(frame.qValues, frame.behaviorPolicy, config);
      }
      return { episode: i + 1, residual };
    });
  }, [result, config, algorithm]);

  const predictionPanelData = useMemo(() => {
    if (!isPrediction) return [];
    let vPi: number[];
    try {
      vPi = solveStateValues(policy, config);
    } catch {
      vPi = new Array(numStates).fill(0);
    }
    return result.frames.slice(1).map((frame, i) => {
      const f = frame as PredictionFrame;
      const rmse = Math.sqrt(f.values.reduce((sum, v, s) => sum + (v - vPi[s]) ** 2, 0) / f.values.length);
      return {
        episode: i + 1,
        vStart: f.values[config.startState],
        rmse,
        residual: policyBellmanResidualV(f.values, f.policy, config),
      };
    });
  }, [result, isPrediction, policy, config, numStates]);

  const controlPanelData = useMemo(() => {
    if (isPrediction) return [];
    let qStar: number[][];
    try {
      qStar = estimateTrueActionValues(config);
    } catch {
      qStar = Array.from({ length: numStates }, () => new Array(5).fill(0));
    }
    return Array.from({ length: episodes }, (_, ep) => {
      const epUpdates = result.updates.filter((u) => u.episode === ep);
      const ret = epUpdates.reduce((sum, u) => sum + u.reward, 0);
      const length = epUpdates.length;
      const frame = result.frames[ep + 1] as ControlFrame;
      const residual = algorithm === 'qlearning'
        ? optimalBellmanResidualQ(frame.qValues, config)
        : policyBellmanResidualQ(frame.qValues, frame.behaviorPolicy, config);
      const greedyAgreement = computeGreedyAgreement(frame.qValues, qStar, config);
      return { episode: ep + 1, return: ret, length, residual, greedyAgreement };
    });
  }, [result, isPrediction, config, algorithm, episodes, numStates]);

  const currentEpisode = currentUpdate?.episode ?? (currentFrame ? safeStep : 0);
  const currentEpsilon = isPrediction ? null : epsilonAtEpisode(currentEpisode, epsilon, epsilonMode);

  return (
    <div className="space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Clock className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">{title}</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">{subtitle}</p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <InteractiveDemo title="TD 算法动态教学">
        <div className="grid lg:grid-cols-[1fr_360px] gap-6">
          <div className="space-y-4">
            <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
              <GridWorld
                config={config}
                policy={display.policy}
                values={display.values}
                showValues
                highlightState={display.highlightState}
                highlightNextState={display.highlightNextState}
                highlightUpdatedState={display.highlightUpdatedState}
                highlightAction={display.highlightAction}
                className="max-w-full"
              />
              <p className="mt-4 text-sm text-gray-500 text-center">
                {viewMode === 'transition'
                  ? `transition ${safeStep + 1} / ${result.updates.length}`
                  : `episode ${safeStep} / ${totalSteps}`}
                {' '}· {display.valueLabel}
                {currentEpsilon !== null && (
                  <span className="ml-2 text-xs text-gray-400">
                    ε = {currentEpsilon.toFixed(3)}
                  </span>
                )}
              </p>
            </div>

            {currentUpdate && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">当前转移与更新</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div className="bg-gray-50 rounded p-2 text-center">
                      <div className="text-gray-500">S_t</div>
                      <div className="font-mono font-semibold">s{currentUpdate.state + 1}</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2 text-center">
                      <div className="text-gray-500">A_t</div>
                      <div className="font-mono font-semibold">{actionName(currentUpdate.action)}</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2 text-center">
                      <div className="text-gray-500">R_{'{t+1}'}</div>
                      <div className="font-mono font-semibold">{formatVal(currentUpdate.reward)}</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2 text-center">
                      <div className="text-gray-500">S_{'{t+1}'}</div>
                      <div className="font-mono font-semibold">s{currentUpdate.nextState + 1}</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2 text-center">
                      <div className="text-gray-500">更新前</div>
                      <div className="font-mono font-semibold">{formatVal(currentUpdate.oldEstimate)}</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2 text-center">
                      <div className="text-gray-500">bootstrap</div>
                      <div className="font-mono font-semibold">{formatVal(currentUpdate.bootstrapValue)}</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2 text-center">
                      <div className="text-gray-500">target</div>
                      <div className="font-mono font-semibold">{formatVal(currentUpdate.target)}</div>
                    </div>
                    <div className="bg-gray-50 rounded p-2 text-center">
                      <div className="text-gray-500">TD error</div>
                      <div className="font-mono font-semibold">{formatVal(currentUpdate.tdError)}</div>
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                    <UpdateFormula
                      update={currentUpdate}
                      algorithm={algorithm}
                      alpha={alpha}
                      gamma={config.gamma}
                    />
                  </div>
                  {currentUpdate.done && (
                    <p className="text-sm text-green-700">该转移到达 terminal 状态，target 仅包含即时奖励。</p>
                  )}
                </CardContent>
              </Card>
            )}
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">选择算法</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-2">
                {algorithms.map(({ key, label }) => (
                  <Button
                    key={key}
                    variant={algorithm === key ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setAlgorithm(key)}
                  >
                    {label}
                  </Button>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">视图与价值显示</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant={viewMode === 'transition' ? 'default' : 'outline'}
                    className="flex-1"
                    onClick={() => setViewMode('transition')}
                  >
                    逐 transition
                  </Button>
                  <Button
                    size="sm"
                    variant={viewMode === 'episode' ? 'default' : 'outline'}
                    className="flex-1"
                    onClick={() => setViewMode('episode')}
                  >
                    按 episode
                  </Button>
                </div>
                {!isPrediction && (
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant={valueMode === 'behavior' ? 'default' : 'outline'}
                      className="flex-1"
                      onClick={() => setValueMode('behavior')}
                    >
                      行为策略价值
                    </Button>
                    <Button
                      size="sm"
                      variant={valueMode === 'greedy' ? 'default' : 'outline'}
                      className="flex-1"
                      onClick={() => setValueMode('greedy')}
                    >
                      贪心派生价值
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">任务与策略</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <div className="text-sm text-gray-600 mb-1">任务类型</div>
                  <Select value={task} onValueChange={(v) => setTask(v as 'continuing' | 'episodic')}>
                    <SelectTrigger className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="continuing">continuing textbook task</SelectItem>
                      <SelectItem value="episodic">episodic path-finding task</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-gray-500 mt-1">
                    {isPrediction
                      ? '预测任务默认使用长度为 H 的训练轨迹。'
                      : '控制任务默认为到达目标或达到最大长度 H 后结束的 episode。'}
                  </p>
                </div>

                {isPrediction && (
                  <div>
                    <div className="text-sm text-gray-600 mb-1">策略预设</div>
                    <Select value={policyPreset} onValueChange={(v) => setPolicyPreset(v as 'goal' | 'random' | 'right')}>
                      <SelectTrigger className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="goal">通向目标的确定性策略</SelectItem>
                        <SelectItem value="random">uniform random policy</SelectItem>
                        <SelectItem value="right">全向右反例策略</SelectItem>
                      </SelectContent>
                    </Select>
                    {policyPreset === 'right' && (
                      <p className="text-xs text-amber-600 mt-1">反例：在右边界反复碰撞。</p>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">超参数</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="text-sm text-gray-600 mb-1">学习率 α</div>
                  <Slider value={[alpha]} min={0.01} max={0.5} step={0.01} onValueChange={([v]) => setAlpha(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{alpha.toFixed(2)}</div>
                </div>

                {!isPrediction && (
                  <>
                    <div>
                      <div className="text-sm text-gray-600 mb-1">探索率 ε</div>
                      <Slider value={[epsilon]} min={0} max={1} step={0.05} onValueChange={([v]) => setEpsilon(v)} />
                      <div className="mt-1 text-center font-mono text-sm text-gray-700">{epsilon.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600 mb-1">ε 调度</div>
                      <Select value={epsilonMode} onValueChange={(v) => setEpsilonMode(v as EpsilonScheduleMode)}>
                        <SelectTrigger className="w-full">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="fixed">fixed epsilon</SelectItem>
                          <SelectItem value="decay-floor">decay with floor</SelectItem>
                          <SelectItem value="glie">GLIE</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </>
                )}

                {(algorithm === 'td-lambda' || algorithm === 'sarsa-lambda') && (
                  <div>
                    <div className="text-sm text-gray-600 mb-1">资格迹衰减 λ</div>
                    <Slider value={[lambda]} min={0} max={0.99} step={0.01} onValueChange={([v]) => setLambda(v)} />
                    <div className="mt-1 text-center font-mono text-sm text-gray-700">{lambda.toFixed(2)}</div>
                  </div>
                )}

                {algorithm === 'nstep' && (
                  <div>
                    <div className="text-sm text-gray-600 mb-1">步数 n</div>
                    <Slider value={[nStep]} min={1} max={10} step={1} onValueChange={([v]) => setNStep(v)} />
                    <div className="mt-1 text-center font-mono text-sm text-gray-700">{nStep}</div>
                  </div>
                )}

                <div>
                  <div className="text-sm text-gray-600 mb-1">轨迹/episode 长度 H</div>
                  <Select value={String(horizonH)} onValueChange={(v) => setHorizonH(Number(v))}>
                    <SelectTrigger className="w-full">
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

                <div>
                  <div className="text-sm text-gray-600 mb-1">训练回合数</div>
                  <Slider value={[episodes]} min={10} max={300} step={10} onValueChange={([v]) => setEpisodes(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{episodes}</div>
                </div>

                <div>
                  <div className="text-sm text-gray-600 mb-1">随机种子</div>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(Number(e.target.value))}
                      className="flex-1 border rounded px-2 py-1 text-sm"
                    />
                    <Button variant="outline" size="sm" onClick={() => setSeed(Math.floor(Math.random() * 100000))}>
                      <RefreshCw className="w-4 h-4 mr-1" />
                      重新生成
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">回放控制</CardTitle>
              </CardHeader>
              <CardContent>
                <AlgorithmPlayer maxStep={totalSteps} currentStep={safeStep} onStepChange={setStep} />
              </CardContent>
            </Card>
          </div>
        </div>
      </InteractiveDemo>

      <InteractiveDemo title="逐 transition TD 误差">
        <LineChart
          data={transitionErrorData}
          xKey="time"
          xLabel="transition"
          yLabel="δ"
          series={[{ key: 'tdError', name: 'sample TD error', color: '#8b5cf6' }]}
        />
      </InteractiveDemo>

      <InteractiveDemo title="按 episode Bellman 残差">
        <LineChart
          data={residualData}
          xKey="episode"
          xLabel="episode"
          yLabel={
            isPrediction
              ? '‖TπV - V‖∞'
              : algorithm === 'qlearning'
                ? '‖T*Q - Q‖∞'
                : '‖TπQ - Q‖∞'
          }
          series={[{ key: 'residual', name: 'Bellman residual', color: '#ef4444' }]}
        />
        <p className="mt-2 text-sm text-gray-600">
          Bellman residual 通常会随学习改善，但在固定学习率、随机采样和控制策略持续变化时可能波动，不保证逐回合单调下降。
        </p>
      </InteractiveDemo>

      {isPrediction && (
        <InteractiveDemo title="TD prediction 面板">
          <LineChart
            data={predictionPanelData}
            xKey="episode"
            xLabel="episode"
            yLabel="价值 / RMSE / 残差"
            series={[
              { key: 'vStart', name: 'V(s_start)', color: '#2563eb' },
              { key: 'rmse', name: 'RMSE(V, V_π)', color: '#22c55e' },
              { key: 'residual', name: 'policy Bellman residual', color: '#ef4444' },
            ]}
          />
        </InteractiveDemo>
      )}

      {!isPrediction && (
        <InteractiveDemo title={`${algorithmDisplay(algorithm)} 控制面板`}>
          <div className="space-y-6">
            <LineChart
              data={controlPanelData}
              xKey="episode"
              xLabel="episode"
              yLabel="回报 / 长度"
              series={[
                { key: 'return', name: task === 'episodic' ? 'episode return' : '长度 H 的截断轨迹累计奖励', color: '#2563eb' },
                { key: 'length', name: task === 'episodic' ? 'episode length' : '截断轨迹长度', color: '#f59e0b' },
              ]}
            />
            <LineChart
              data={controlPanelData}
              xKey="episode"
              xLabel="episode"
              yLabel="残差 / 一致率"
              series={[
                { key: 'residual', name: algorithm === 'qlearning' ? 'Bellman optimality residual' : 'behavior-policy Bellman residual', color: '#ef4444' },
                { key: 'greedyAgreement', name: 'greedy action agreement', color: '#22c55e' },
              ]}
            />
          </div>
        </InteractiveDemo>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">收敛条件（随机逼近要求）</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc pl-5 space-y-1 text-sm text-gray-700">
            <li>充分访问所有 state-action pair；</li>
            <li>Σ_t α_t = ∞；</li>
            <li>Σ_t α_t² &lt; ∞；</li>
            <li>对控制问题采用适当的探索衰减（如 GLIE）。</li>
          </ul>
          <p className="mt-3 text-sm text-gray-600">
            在有限 MDP、充分探索和满足随机逼近条件的递减步长下，Q-learning 可收敛到 q*。Sarsa 控制还需要相应的 GLIE 和步长条件。本页面使用有限样本和可调参数，主要展示更新机制，不等同于渐近收敛证明。
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function algorithmDisplay(a: AlgorithmKind): string {
  switch (a) {
    case 'td0':
      return 'TD(0)';
    case 'sarsa':
      return 'Sarsa';
    case 'nstep':
      return 'n-step Sarsa';
    case 'qlearning':
      return 'Q-learning';
    case 'expected':
      return 'Expected Sarsa';
    case 'sarsa-lambda':
      return 'Sarsa(λ)';
    case 'td-lambda':
      return 'TD(λ)';
  }
}
