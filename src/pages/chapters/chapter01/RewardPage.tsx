import { useState, useMemo } from 'react';
import { Gift, AlertTriangle, Scale, ShieldAlert, Play, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import {
  DEFAULT_CONFIG,
  ACTION_NAMES,
  type Action,
  type Policy,
  deterministicPolicy,
  nextState,
  reward,
  rewardForAction,
  isTerminal,
} from '@/lib/rl/gridworld';

export default function Chapter01RewardPage() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [offset, setOffset] = useState(0);
  const [showOffset, setShowOffset] = useState(false);
  const [trajectory, setTrajectory] = useState<number[]>([DEFAULT_CONFIG.startState]);
  const [message, setMessage] = useState('调节奖励参数，观察即时奖励表与贪心策略的关系。');

  const effectiveConfig = useMemo(
    () => ({
      ...config,
      targetReward: config.targetReward + (showOffset ? offset : 0),
      forbiddenReward: config.forbiddenReward + (showOffset ? offset : 0),
      boundaryReward: config.boundaryReward + (showOffset ? offset : 0),
      stepReward: config.stepReward + (showOffset ? offset : 0),
    }),
    [config, offset, showOffset]
  );

  const stateRewards = useMemo(
    () => Array.from({ length: 9 }, (_, s) => reward(s, effectiveConfig)),
    [effectiveConfig]
  );

  const immediateGreedyPolicy = useMemo<Policy>(() => {
    const actions: Action[] = Array.from({ length: 9 }, (_, s) => {
      let bestA: Action = 4;
      let bestR = -Infinity;
      for (let a = 0; a < 5; a++) {
        const r = rewardForAction(s, a as Action, effectiveConfig);
        if (r > bestR) {
          bestR = r;
          bestA = a as Action;
        }
      }
      return bestA;
    });
    return deterministicPolicy(actions);
  }, [effectiveConfig]);

  function runGreedyEpisode() {
    let state = effectiveConfig.startState;
    const traj = [state];
    const maxSteps = 12;
    let reachedTerminal = false;

    for (let step = 0; step < maxSteps; step++) {
      if (isTerminal(state, effectiveConfig)) break;
      const action = immediateGreedyPolicy[state].indexOf(1) as Action;
      const sNext = nextState(state, action, effectiveConfig);
      traj.push(sNext);
      state = sNext;
      if (isTerminal(state, effectiveConfig)) {
        reachedTerminal = true;
        break;
      }
    }

    setTrajectory(traj);
    setMessage(
      reachedTerminal
        ? '按即时奖励贪心策略到达目标。注意：即时奖励贪心未必总能找到最优路径。'
        : '未到达目标，可能陷入循环或禁区/边界。说明仅看即时奖励会短视。'
    );
  }

  function rewardClass(r: number): string {
    if (r > 0) return 'text-green-700 bg-green-50';
    if (r < 0) return 'text-red-700 bg-red-50';
    return 'text-gray-700 bg-gray-50';
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      {/* Hero */}
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-rose-100 rounded-xl flex items-center justify-center">
            <Gift className="w-8 h-8 text-rose-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          1.3 奖励设计
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          奖励是强化学习中的人机接口。理解即时奖励、奖励的相对性，以及为什么只看即时奖励会短视。
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
              <Gift className="w-4 h-4" />
              奖励是人机接口
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            正奖励鼓励某个动作，负奖励抑制某个动作。
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              即时奖励陷阱
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            当前奖励最大的动作不一定带来最大长期回报。
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Scale className="w-4 h-4" />
              奖励的相对性
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            对所有奖励加同一常数，贪心动作排序不变。
          </CardContent>
        </Card>
      </section>

      {/* Interactive demo */}
      <InteractiveDemo title="奖励设计工作台">
        <div className="grid lg:grid-cols-[1fr_380px] gap-6">
          <div className="space-y-6">
            <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
              <GridWorld
                config={effectiveConfig}
                values={stateRewards}
                showValues
                trajectory={trajectory}
                currentStep={trajectory.length - 1}
                className="max-w-full"
              />
              <p className="mt-4 text-sm text-gray-500 text-center">
                格子颜色与数值表示该状态的即时奖励 r(s)
              </p>
            </div>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">即时奖励贪心策略</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col sm:flex-row gap-4 items-start">
                  <GridWorld
                    config={effectiveConfig}
                    policy={immediateGreedyPolicy}
                    className="max-w-[220px]"
                  />
                  <div className="space-y-3 flex-1">
                    <p className="text-sm text-gray-600">
                      每个状态的箭头表示<strong>即时奖励最大</strong>的动作。点击「运行贪心一回合」看看它会走到哪里。
                    </p>
                    <div className="flex gap-2">
                      <Button size="sm" onClick={runGreedyEpisode} className="bg-rose-600 hover:bg-rose-700">
                        <Play className="w-4 h-4 mr-1" />
                        运行贪心一回合
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => {
                          setTrajectory([effectiveConfig.startState]);
                          setMessage('已重置');
                        }}
                      >
                        <RotateCcw className="w-4 h-4 mr-1" />
                        重置
                      </Button>
                    </div>
                    <div className="text-sm text-gray-700 bg-rose-50 p-3 rounded-lg border border-rose-100">
                      {message}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">奖励参数</CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1">
                  <span>目标奖励 r_target</span>
                  <span className="font-mono">{config.targetReward.toFixed(1)}</span>
                </div>
                <Slider
                  value={[config.targetReward]}
                  min={-5}
                  max={5}
                  step={0.5}
                  onValueChange={([v]) => setConfig((prev) => ({ ...prev, targetReward: v }))}
                />
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1">
                  <span>禁区奖励 r_forbidden</span>
                  <span className="font-mono">{config.forbiddenReward.toFixed(1)}</span>
                </div>
                <Slider
                  value={[config.forbiddenReward]}
                  min={-5}
                  max={5}
                  step={0.5}
                  onValueChange={([v]) => setConfig((prev) => ({ ...prev, forbiddenReward: v }))}
                />
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1">
                  <span>边界奖励 r_boundary</span>
                  <span className="font-mono">{config.boundaryReward.toFixed(1)}</span>
                </div>
                <Slider
                  value={[config.boundaryReward]}
                  min={-5}
                  max={5}
                  step={0.5}
                  onValueChange={([v]) => setConfig((prev) => ({ ...prev, boundaryReward: v }))}
                />
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1">
                  <span>普通步奖励 r_other</span>
                  <span className="font-mono">{config.stepReward.toFixed(1)}</span>
                </div>
                <Slider
                  value={[config.stepReward]}
                  min={-5}
                  max={5}
                  step={0.5}
                  onValueChange={([v]) => setConfig((prev) => ({ ...prev, stepReward: v }))}
                />
              </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">奖励的相对性</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">给所有奖励加同一偏移量</span>
                  <Switch checked={showOffset} onCheckedChange={setShowOffset} />
                </div>
                {showOffset && (
                  <>
                    <Slider
                      value={[offset]}
                      min={-3}
                      max={3}
                      step={0.5}
                      onValueChange={([v]) => setOffset(v)}
                    />
                    <div className="text-center font-mono text-sm text-gray-700">
                      offset = {offset.toFixed(1)}
                    </div>
                  </>
                )}
                <p className="text-sm text-gray-600">
                  无论偏移量如何，每个状态下即时奖励最大的动作集合保持不变。最优策略对奖励的仿射变换保持不变。
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">即时奖励表 r(s,a)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto max-h-[320px] overflow-y-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead className="sticky top-0 bg-gray-50">
                      <tr className="border-b border-gray-200 text-gray-600">
                        <th className="py-1 pr-2 text-left">状态</th>
                        {ACTION_NAMES.map((name) => (
                          <th key={name} className="py-1 pr-2 text-left">
                            {name}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Array.from({ length: 9 }, (_, s) => (
                        <tr key={s} className="border-b border-gray-100">
                          <td className="py-1 pr-2 font-medium">s{s + 1}</td>
                          {ACTION_NAMES.map((_, a) => {
                            const r = rewardForAction(s, a as Action, effectiveConfig);
                            return (
                              <td key={a} className="py-1 pr-2">
                                <span
                                  className={`inline-block px-1.5 py-0.5 rounded text-xs font-mono ${rewardClass(
                                    r
                                  )}`}
                                >
                                  {r.toFixed(1)}
                                </span>
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </InteractiveDemo>

      {/* Formula summary */}
      <section className="space-y-4">
        <FormulaCard
          title="奖励函数"
          formula={<KaTeX math={String.raw`r(s,a) = \mathbb{E}[R_{t+1} \mid S_t=s, A_t=a]`} display />}
          description="在本书网格世界中，r(s,a) 由执行动作后到达的下一状态决定。"
        />
        <FormulaCard
          title="奖励的仿射不变性"
          formula={<KaTeX math={String.raw`r'(s,a) = \alpha r(s,a) + \beta, \quad \alpha > 0`} display />}
          description="对奖励进行正的缩放与平移不改变最优策略。"
        />
      </section>
    </div>
  );
}
