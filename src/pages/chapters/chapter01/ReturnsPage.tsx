import { useState, useMemo } from 'react';
import { Clock, Infinity, Dices, ShieldAlert, Play, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import AlgorithmPlayer from '@/components/AlgorithmPlayer';
import TrajectoryViewer from '@/components/TrajectoryViewer';
import LineChart from '@/components/LineChart';
import SeedControl from '@/components/SeedControl';
import { mulberry32 } from '@/lib/rl/stochasticApproximation';
import {
  DEFAULT_CONFIG,
  ACTION_NAMES,
  type Action,
  type Policy,
  deterministicPolicy,
  step,
  isTerminal,
  discountedReturn,
  sampleActionWithRng,
} from '@/lib/rl/gridworld';

export default function Chapter01ReturnsPage() {
  const config = DEFAULT_CONFIG;
  const [policy, setPolicy] = useState<Policy>(() =>
    deterministicPolicy([1, 1, 2, 1, 1, 2, 4, 1, 4] as Action[])
  );
  const [startState, setStartState] = useState(config.startState);
  const [gamma, setGamma] = useState(config.gamma);
  const [trajectory, setTrajectory] = useState<
    { state: number; action: Action; reward: number; nextState: number }[]
  >([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [absorbing, setAbsorbing] = useState(false);
  const [markovGuess, setMarkovGuess] = useState<number | null>(null);
  const [seed, setSeed] = useState(1);

  function runEpisode() {
    let state = startState;
    const traj: { state: number; action: Action; reward: number; nextState: number }[] = [];
    const maxSteps = 15;
    const rng = mulberry32(seed);

    for (let stepIdx = 0; stepIdx < maxSteps; stepIdx++) {
      if (isTerminal(state, config)) {
        if (absorbing) {
          // absorbing target: stay forever with a5
          const result = step(state, 4, config);
          traj.push({ state, action: 4, reward: result.reward, nextState: result.nextState });
          if (traj.length >= maxSteps) break;
        } else {
          break;
        }
      } else {
        const action = sampleActionWithRng(policy[state], rng) as Action;
        const result = step(state, action, config);
        traj.push({ state, action, reward: result.reward, nextState: result.nextState });
        state = result.nextState;
      }
    }

    setTrajectory(traj);
    setCurrentStep(0);
  }

  const returnValue = useMemo(
    () => (trajectory.length > 0 ? discountedReturn(trajectory, gamma) : 0),
    [trajectory, gamma]
  );

  const returnCurve = useMemo(() => {
    if (trajectory.length === 0) return [];
    const data: Record<string, number>[] = [];
    for (let g = 0; g <= 0.99; g += 0.03) {
      data.push({ gamma: Number(g.toFixed(2)), return: discountedReturn(trajectory, g) });
    }
    return data;
  }, [trajectory]);

  const infiniteReturn = useMemo(
    () => (absorbing && config.targetReward !== 0 ? config.targetReward / (1 - gamma) : null),
    [absorbing, config.targetReward, gamma]
  );

  const trajectoryStates = useMemo(
    () => [startState, ...trajectory.map((t) => t.nextState)],
    [trajectory, startState]
  );

  // Markov quiz: deterministic next state for s5 + a2 (right) is s6
  const quizState = 4;
  const quizAction = 1;
  const quizNext = step(quizState, quizAction, config).nextState;
  const quizHistory = 's1→a2→s2→a3→s5';
  const markovCorrect = markovGuess === quizNext;

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      {/* Hero */}
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-emerald-100 rounded-xl flex items-center justify-center">
            <Clock className="w-8 h-8 text-emerald-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          1.4 回报、折扣与马尔可夫性
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          从有限轨迹的累积奖励，到无限轨迹的折扣回报；从回合任务到持续任务，再到马尔可夫决策过程的形式化。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      {/* Concept cards */}
      <section className="grid md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Clock className="w-4 h-4" />
              折扣回报
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            <KaTeX math={String.raw`G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}`} />
            γ 越大越重视远期奖励。
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Infinity className="w-4 h-4" />
              回合与持续任务
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            到达目标即停止是回合任务；把目标视为普通状态则可持续运行，需要折扣因子保证收敛。
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Dices className="w-4 h-4" />
              马尔可夫性
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            <KaTeX math={String.raw`p(s_{t+1}|s_t,a_t,\dots) = p(s_{t+1}|s_t,a_t)`} />
            下一状态只依赖当前状态与动作。
          </CardContent>
        </Card>
      </section>

      {/* Interactive demo: discounted return */}
      <InteractiveDemo title="折扣回报可视化">
        <div className="grid lg:grid-cols-[1fr_380px] gap-6">
          <div className="space-y-6">
            <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
              <GridWorld
                config={config}
                policy={policy}
                trajectory={trajectoryStates}
                currentStep={currentStep}
                highlightState={trajectoryStates[currentStep] ?? null}
                onCellClick={(s) => setStartState(s)}
                onActionClick={(s, a) => {
                  setPolicy((prev) =>
                    prev.map((dist, state) => {
                      if (state !== s) return dist;
                      const d = new Array(5).fill(0);
                      d[a] = 1;
                      return d;
                    })
                  );
                }}
                editable
                className="max-w-full"
              />
              <p className="mt-4 text-sm text-gray-500 text-center">
                点击格子设置起始状态；点击箭头编辑策略
              </p>
            </div>

            {trajectory.length > 0 && (
              <TrajectoryViewer
                steps={trajectory.map((t, i) => ({
                  t: i,
                  state: `s${t.state + 1}`,
                  action: ACTION_NAMES[t.action],
                  reward: t.reward,
                  nextState: `s${t.nextState + 1}`,
                }))}
                currentIndex={currentStep}
              />
            )}
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">生成轨迹</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="text-sm text-gray-700 mb-2">起始状态</div>
                  <div className="flex flex-wrap gap-1">
                    {Array.from({ length: 9 }, (_, i) => (
                      <button
                        key={i}
                        onClick={() => setStartState(i)}
                        className={`w-8 h-8 rounded text-xs font-medium border transition-colors ${
                          startState === i
                            ? 'bg-emerald-600 text-white border-emerald-600'
                            : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
                        }`}
                      >
                        s{i + 1}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">目标为吸收态</span>
                  <input
                    type="checkbox"
                    checked={absorbing}
                    onChange={(e) => setAbsorbing(e.target.checked)}
                    className="w-5 h-5 rounded border-gray-300 text-emerald-600 focus:ring-emerald-500"
                  />
                </div>

                <SeedControl seed={seed} onChange={setSeed} />

                <div className="grid grid-cols-2 gap-2">
                  <Button size="sm" onClick={runEpisode} className="bg-emerald-600 hover:bg-emerald-700">
                    <Play className="w-4 h-4 mr-1" />
                    运行一回合
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      setTrajectory([]);
                      setCurrentStep(0);
                    }}
                  >
                    <RotateCcw className="w-4 h-4 mr-1" />
                    清空
                  </Button>
                </div>

                {trajectory.length > 0 && (
                  <AlgorithmPlayer
                    maxStep={trajectory.length - 1}
                    currentStep={currentStep}
                    onStepChange={setCurrentStep}
                  />
                )}

                {trajectory.length > 0 && (
                  <div className="text-sm text-gray-700 bg-emerald-50 p-3 rounded-lg border border-emerald-100">
                    折扣回报（γ={gamma.toFixed(2)}）：
                    <span className="font-mono font-semibold ml-1">
                      {returnValue.toFixed(3)}
                    </span>
                  </div>
                )}

                {infiniteReturn !== null && (
                  <div className="text-sm text-gray-700">
                    持续停留在目标的无限回报：
                    <KaTeX math={String.raw`\frac{r}{1-\gamma} = ${infiniteReturn.toFixed(2)}`} />
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">折扣因子 γ</CardTitle>
              </CardHeader>
              <CardContent>
                <Slider value={[gamma]} min={0} max={0.99} step={0.01} onValueChange={([v]) => setGamma(v)} />
                <div className="mt-2 text-center font-mono text-sm text-gray-700">
                  γ = {gamma.toFixed(2)}
                </div>
                <p className="mt-2 text-xs text-gray-500">
                  γ 接近 0 时短视，接近 1 时重视远期奖励。
                </p>
              </CardContent>
            </Card>
          </div>
        </div>

        {returnCurve.length > 0 && (
          <div className="mt-6">
            <LineChart
              data={returnCurve}
              xKey="gamma"
              xLabel="折扣因子 γ"
              yLabel="折扣回报 G"
              series={[{ key: 'return', name: '折扣回报', color: '#059669' }]}
              height={260}
            />
          </div>
        )}
      </InteractiveDemo>

      {/* Markov quiz */}
      <InteractiveDemo title="马尔可夫性小测验">
        <div className="grid md:grid-cols-2 gap-6 items-center">
          <div className="space-y-4">
            <p className="text-gray-700">
              已知完整历史：{quizHistory}。当前处于状态 <strong>s{quizState + 1}</strong>，准备执行动作
              <strong> {ACTION_NAMES[quizAction]}</strong>。
            </p>
            <p className="text-gray-700">
              请问：在本课程的确定性网格世界中，仅知道 <strong>(s{quizState + 1}, {ACTION_NAMES[quizAction]})</strong> 与知道完整历史，预测的下一状态是否相同？
            </p>
            <div className="flex flex-wrap gap-2">
              {Array.from({ length: 9 }, (_, i) => (
                <button
                  key={i}
                  onClick={() => setMarkovGuess(i)}
                  className={`px-3 py-1.5 rounded text-sm font-medium border transition-colors ${
                    markovGuess === i
                      ? 'bg-emerald-600 text-white border-emerald-600'
                      : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
                  }`}
                >
                  s{i + 1}
                </button>
              ))}
            </div>
            {markovGuess !== null && (
              <div
                className={`text-sm p-3 rounded-lg border ${
                  markovCorrect
                    ? 'bg-green-50 text-green-800 border-green-200'
                    : 'bg-red-50 text-red-800 border-red-200'
                }`}
              >
                {markovCorrect
                  ? '正确！这正是马尔可夫性：未来只依赖当前状态与动作，与历史无关。'
                  : `不正确。确定性转移下，s${quizState + 1} 向右一定到达 s${quizNext + 1}。`}
              </div>
            )}
          </div>
          <div className="flex justify-center">
            <GridWorld
              config={config}
              highlightState={quizState}
              className="max-w-[240px]"
            />
          </div>
        </div>
      </InteractiveDemo>

      {/* Formula summary */}
      <section className="space-y-4">
        <FormulaCard
          title="折扣回报"
          formula={<KaTeX math={String.raw`G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}`} display />}
          description="折扣因子让无限轨迹的回报收敛，并调节对远期奖励的重视程度。"
        />
        <FormulaCard
          title="马尔可夫性"
          formula={
            <KaTeX
              math={String.raw`p(s_{t+1}|s_t,a_t,\dots,s_0,a_0) = p(s_{t+1}|s_t,a_t)`}
              display
            />
          }
          description="下一状态与奖励只由当前状态和动作决定，与历史无关。"
        />
      </section>
    </div>
  );
}
