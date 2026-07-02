import { useState, useMemo } from 'react';
import { Grid3x3, Play, RotateCcw, Info, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
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
  isTerminal,
} from '@/lib/rl/gridworld';

export default function Chapter01MdpPage() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [policy, setPolicy] = useState<Policy>(() =>
    deterministicPolicy([1, 1, 3, 1, 1, 1, 1, 1, 4] as Action[])
  );
  const [agentState, setAgentState] = useState(config.startState);
  const [trajectory, setTrajectory] = useState<number[]>([config.startState]);
  const [selectedState, setSelectedState] = useState<number | null>(null);
  const [message, setMessage] = useState('点击「运行一回合」开始交互演示');

  const selectedInfo = useMemo(() => {
    if (selectedState === null) return null;
    return {
      state: selectedState,
      reward: reward(selectedState, config),
      terminal: isTerminal(selectedState, config),
    };
  }, [selectedState, config]);

  const MAX_EPISODE_STEPS = 12;

  function runEpisode() {
    let state = config.startState;
    const traj = [state];
    let msg = `从 s${state + 1} 出发`;
    let reachedTerminal = false;

    for (let step = 0; step < MAX_EPISODE_STEPS; step++) {
      if (isTerminal(state, config)) break;
      const actionDist = policy[state];
      const action = sampleAction(actionDist) as Action;
      const sNext = nextState(state, action, config);
      const r = reward(sNext, config);
      traj.push(sNext);
      msg += ` → ${ACTION_NAMES[action]} → s${sNext + 1}(r=${r})`;
      state = sNext;
      if (isTerminal(state, config)) {
        msg += ' [到达目标]';
        reachedTerminal = true;
        break;
      }
      if (config.forbiddenStates.includes(state)) {
        msg += ' [进入禁区]';
      }
    }

    if (!reachedTerminal && !isTerminal(state, config)) {
      msg += ` [已达最大步数 ${MAX_EPISODE_STEPS}，未到达目标]`;
    }

    setAgentState(state);
    setTrajectory(traj);
    setMessage(msg);
  }

  function stepOnce() {
    if (isTerminal(agentState, config)) {
      setMessage(`已在终止状态 s${agentState + 1}，请重置后再执行`);
      return;
    }
    const actionDist = policy[agentState];
    const action = sampleAction(actionDist) as Action;
    const sNext = nextState(agentState, action, config);
    const r = reward(sNext, config);
    const newTrajectory = [...trajectory, sNext];
    let msg = `从 s${agentState + 1} ${ACTION_NAMES[action]} 到 s${sNext + 1}，奖励 ${r}`;
    if (isTerminal(sNext, config)) msg += ' [到达目标]';
    else if (config.forbiddenStates.includes(sNext)) msg += ' [进入禁区]';

    setAgentState(sNext);
    setTrajectory(newTrajectory);
    setMessage(msg);
  }

  function reset() {
    setAgentState(config.startState);
    setTrajectory([config.startState]);
    setMessage('已重置');
  }

  function updatePolicy(state: number, action: number) {
    const newPolicy = deterministicPolicy(
      policy.map((dist, s) => (s === state ? (action as Action) : (dist.indexOf(1) as Action))) as Action[]
    );
    setPolicy(newPolicy);
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      {/* Hero */}
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Grid3x3 className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          1.1 马尔可夫决策过程
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          通过网格世界交互理解马尔可夫决策过程的五个核心要素：状态、动作、转移、奖励与策略。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      {/* Concept cards */}
      <section className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">状态</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            智能体在环境中的位置。本例中有 9 个状态：
            <KaTeX math={String.raw`\mathcal{S} = \{s_1, \dots, s_9\}`} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">动作</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            每个状态有 5 个可选动作：
            <KaTeX math={String.raw`\mathcal{A} = \{a_1(\text{上}), a_2(\text{右}), a_3(\text{下}), a_4(\text{左}), a_5(\text{停留})\}`} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">状态转移</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            确定性转移：
            <KaTeX math={String.raw`p(s'|s,a) = 1 \text{ if } s' = f(s,a)`} />
            撞边界则反弹回当前格。
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">奖励</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            目标格 +1，禁区 -1，撞边界 -1，普通步 0。
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">策略</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            <KaTeX math={String.raw`\pi(a|s)`} /> 表示在状态 <KaTeX math={String.raw`s`} /> 选择动作 <KaTeX math={String.raw`a`} /> 的概率。
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">回报</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            <KaTeX math={String.raw`G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}`} />
            折扣因子 <KaTeX math={String.raw`\gamma \in [0,1)`} />。
          </CardContent>
        </Card>
      </section>

      {/* Interactive demo */}
      <InteractiveDemo title="网格世界交互演示">
        <div className="grid lg:grid-cols-[1fr_340px] gap-6">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld
              config={config}
              policy={policy}
              trajectory={trajectory}
              currentStep={trajectory.length - 1}
              highlightState={agentState}
              onCellClick={setSelectedState}
              onActionClick={updatePolicy}
              editable
              className="max-w-full"
            />
            <p className="mt-4 text-sm text-gray-500 text-center">
              点击网格中的箭头可修改每个状态的策略；使用「单步执行」可逐格观察移动
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Info className="w-4 h-4" />
                  当前信息
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="text-sm text-gray-700 bg-blue-50 p-3 rounded-lg border border-blue-100">
                  {message}
                </div>
                {selectedInfo && (
                  <div className="text-sm text-gray-600">
                    选中状态 <strong>s{selectedInfo.state + 1}</strong>：
                    奖励 {selectedInfo.reward}，
                    {selectedInfo.terminal ? ' 终止状态' : ' 非终止状态'}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">折扣因子 <KaTeX math={String.raw`\gamma`} /></CardTitle>
              </CardHeader>
              <CardContent>
                <Slider
                  value={[config.gamma]}
                  min={0}
                  max={0.99}
                  step={0.01}
                  onValueChange={([v]) => setConfig({ ...config, gamma: v })}
                />
                <div className="mt-2 text-center font-mono text-sm text-gray-700">
                  γ = {config.gamma.toFixed(2)}
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-3 gap-2">
              <Button onClick={stepOnce} size="sm" variant="outline">
                单步执行
              </Button>
              <Button onClick={runEpisode} size="sm" className="bg-blue-600 hover:bg-blue-700">
                <Play className="w-4 h-4 mr-1" />
                运行一回合
              </Button>
              <Button onClick={reset} size="sm" variant="outline">
                <RotateCcw className="w-4 h-4 mr-1" />
                重置
              </Button>
            </div>
          </div>
        </div>
      </InteractiveDemo>

      {/* Formula summary */}
      <section className="space-y-4">
        <FormulaCard
          title="折扣回报"
          formula={<KaTeX math={String.raw`G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}`} display />}
          description="折扣回报衡量从某时刻开始未来奖励的累积价值。"
        />
        <FormulaCard
          title="马尔可夫决策过程"
          formula={<KaTeX math={String.raw`\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)`} display />}
          description="马尔可夫决策过程由状态空间、动作空间、转移概率、奖励函数和折扣因子组成。"
        />
      </section>
    </div>
  );
}

function sampleAction(probs: number[]): number {
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r <= cum) return i;
  }
  return probs.length - 1;
}
