import { useState, useMemo } from 'react';
import { Route, Shuffle, Play, RotateCcw, Target, BookOpen, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import AlgorithmPlayer from '@/components/AlgorithmPlayer';
import TrajectoryViewer from '@/components/TrajectoryViewer';
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

export default function Chapter01PolicyPage() {
  const config = DEFAULT_CONFIG;
  const [policy, setPolicy] = useState<Policy>(() =>
    deterministicPolicy([1, 1, 2, 1, 1, 2, 4, 1, 4] as Action[])
  );
  const [selectedState, setSelectedState] = useState<number>(0);
  const [startState, setStartState] = useState<number>(config.startState);
  const [trajectory, setTrajectory] = useState<
    { state: number; action: Action; reward: number; nextState: number }[]
  >([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [message, setMessage] = useState('编辑策略后运行一回合，观察生成的轨迹与回报。');
  const [seed, setSeed] = useState(1);

  const returnValue = useMemo(
    () => (trajectory.length > 0 ? discountedReturn(trajectory, config.gamma) : 0),
    [trajectory, config.gamma]
  );

  function setDeterministic(state: number, action: Action) {
    setPolicy((prev) =>
      prev.map((dist, s) => {
        if (s !== state) return dist;
        const d = new Array(5).fill(0);
        d[action] = 1;
        return d;
      })
    );
  }

  function setStochasticProb(state: number, action: number, value: number) {
    setPolicy((prev) =>
      prev.map((dist, s) => {
        if (s !== state) return dist;
        const others = dist.map((p, i) => (i === action ? 0 : p));
        const othersSum = others.reduce((a, b) => a + b, 0);
        const newDist = new Array(5).fill(0);
        newDist[action] = value;
        if (othersSum === 0) {
          const uniform = (1 - value) / 4;
          for (let i = 0; i < 5; i++) {
            if (i !== action) newDist[i] = uniform;
          }
        } else {
          for (let i = 0; i < 5; i++) {
            if (i !== action) newDist[i] = (others[i] / othersSum) * (1 - value);
          }
        }
        return newDist;
      })
    );
  }

  function applyPreset(name: 'book' | 'right' | 'random' | 'stay') {
    if (name === 'book') {
      // s1: right/down 0.5; s3, s5: down to avoid forbidden; rest follow right/down heuristic
      const base = deterministicPolicy([1, 1, 2, 1, 1, 2, 4, 1, 4] as Action[]);
      const book = base.map((dist, s) => {
        if (s !== 0) return dist;
        const d = new Array(5).fill(0);
        d[1] = 0.5;
        d[2] = 0.5;
        return d;
      });
      setPolicy(book);
    } else if (name === 'right') {
      setPolicy(deterministicPolicy([1, 1, 1, 1, 1, 1, 1, 1, 4] as Action[]));
    } else if (name === 'stay') {
      setPolicy(deterministicPolicy(new Array(9).fill(4) as Action[]));
    } else {
      setPolicy(
        Array.from({ length: 9 }, () => new Array(5).fill(1 / 5))
      );
    }
  }

  function runEpisode() {
    let state = startState;
    const traj: { state: number; action: Action; reward: number; nextState: number }[] = [];
    const maxSteps = 15;
    const rng = mulberry32(seed);

    for (let stepIdx = 0; stepIdx < maxSteps; stepIdx++) {
      if (isTerminal(state, config)) break;
      const action = sampleActionWithRng(policy[state], rng) as Action;
      const result = step(state, action, config);
      traj.push({ state, action, reward: result.reward, nextState: result.nextState });
      state = result.nextState;
      if (result.done || isTerminal(state, config)) break;
    }

    setTrajectory(traj);
    setCurrentStep(0);
    setMessage(
      traj.length === 0
        ? '起始状态已是终止状态。'
        : `生成 ${traj.length} 步轨迹，折扣回报 ${discountedReturn(traj, config.gamma).toFixed(3)}`
    );
  }

  const trajectoryStates = useMemo(
    () => [startState, ...trajectory.map((t) => t.nextState)],
    [trajectory, startState]
  );

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      {/* Hero */}
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-indigo-100 rounded-xl flex items-center justify-center">
            <Route className="w-8 h-8 text-indigo-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          1.2 策略与轨迹
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          策略决定智能体在每个状态下如何选择动作；按策略执行可生成状态-动作-奖励轨迹。
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
              <Target className="w-4 h-4" />
              确定性策略
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            每个状态只选择一个动作：
            <KaTeX math={String.raw`\pi(a|s) = 1`} /> 对某个 <KaTeX math={String.raw`a`} />。
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Shuffle className="w-4 h-4" />
              随机性策略
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            一个状态下可以按概率分布选择多个动作，如 s1 向右/向下各 0.5。
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <BookOpen className="w-4 h-4" />
              轨迹与回报
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            轨迹是 <KaTeX math={String.raw`s,a,r,s'`} /> 序列；回报是累积奖励。
          </CardContent>
        </Card>
      </section>

      {/* Interactive demo */}
      <InteractiveDemo title="策略与轨迹实验室">
        <div className="grid lg:grid-cols-[1fr_380px] gap-6">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld
              config={config}
              policy={policy}
              trajectory={trajectoryStates}
              currentStep={currentStep}
              highlightState={trajectoryStates[currentStep] ?? null}
              onCellClick={setSelectedState}
              onActionClick={(s, a) => setDeterministic(s, a as Action)}
              editable
              className="max-w-full"
            />
            <p className="mt-4 text-sm text-gray-500 text-center">
              点击箭头设置确定性动作；右侧面板可精细调节概率
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">策略预设</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  <Button size="sm" variant="outline" onClick={() => applyPreset('book')}>
                    书中示例
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => applyPreset('right')}>
                    全部向右
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => applyPreset('random')}>
                    均匀随机
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => applyPreset('stay')}>
                    全部停留
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">
                  编辑状态 s{selectedState + 1} 的策略
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {ACTION_NAMES.map((name, a) => (
                  <div key={a}>
                    <div className="flex justify-between text-sm text-gray-700 mb-1">
                      <span>{name}</span>
                      <span className="font-mono">
                        {policy[selectedState][a].toFixed(2)}
                      </span>
                    </div>
                    <Slider
                      value={[policy[selectedState][a]]}
                      min={0}
                      max={1}
                      step={0.05}
                      onValueChange={([v]) => setStochasticProb(selectedState, a, v)}
                    />
                  </div>
                ))}
                <div className="text-xs text-gray-500">
                  概率之和：{policy[selectedState].reduce((a, b) => a + b, 0).toFixed(2)}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">轨迹控制</CardTitle>
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
                            ? 'bg-blue-600 text-white border-blue-600'
                            : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
                        }`}
                      >
                        s{i + 1}
                      </button>
                    ))}
                  </div>
                </div>

                <SeedControl seed={seed} onChange={setSeed} />

                <div className="grid grid-cols-2 gap-2">
                  <Button onClick={runEpisode} size="sm" className="bg-indigo-600 hover:bg-indigo-700">
                    <Play className="w-4 h-4 mr-1" />
                    运行一回合
                  </Button>
                  <Button
                    onClick={() => {
                      setTrajectory([]);
                      setCurrentStep(0);
                      setMessage('已清空轨迹');
                    }}
                    size="sm"
                    variant="outline"
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

                <div className="text-sm text-gray-700 bg-indigo-50 p-3 rounded-lg border border-indigo-100">
                  {message}
                </div>
                {trajectory.length > 0 && (
                  <div className="text-sm text-gray-700">
                    折扣回报（γ={config.gamma}）：
                    <span className="font-mono font-semibold ml-1">
                      {returnValue.toFixed(3)}
                    </span>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {trajectory.length > 0 && (
          <div className="mt-6">
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
          </div>
        )}
      </InteractiveDemo>

      {/* Formula summary */}
      <section className="space-y-4">
        <FormulaCard
          title="策略"
          formula={<KaTeX math={String.raw`\pi(a|s) = \mathbb{P}(A_t = a \mid S_t = s)`} display />}
          description="策略是一个条件概率分布，决定在每个状态下选择动作的概率。"
        />
        <FormulaCard
          title="折扣回报"
          formula={<KaTeX math={String.raw`G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}`} display />}
          description="从时刻 t 开始，沿着策略生成的轨迹累积未来折扣奖励。"
        />
      </section>
    </div>
  );
}
