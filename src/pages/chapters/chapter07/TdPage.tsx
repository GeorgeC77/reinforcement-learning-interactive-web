import { useState, useMemo } from 'react';
import { Clock, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import {
  DEFAULT_CONFIG,
  deterministicPolicy,
  tdZeroPrediction,
  qLearning,
  sarsa,
  epsilonGreedyPolicy,
  actionValueToStateValue,
} from '@/lib/rl/gridworld';

type Algorithm = 'td0' | 'sarsa' | 'qlearning';

export default function Chapter07TdPage() {
  const [algorithm, setAlgorithm] = useState<Algorithm>('td0');
  const [alpha, setAlpha] = useState(0.2);
  const [epsilon, setEpsilon] = useState(0.3);
  const [episodes, setEpisodes] = useState(100);
  const [step, setStep] = useState(0);

  const config = DEFAULT_CONFIG;

  const history = useMemo(() => {
    if (algorithm === 'td0') {
      const policy = deterministicPolicy([1, 1, 1, 1, 1, 1, 1, 1, 1] as any, 5);
      const values = tdZeroPrediction(policy, config, alpha, episodes);
      return values.map((v) => ({ values: v, policy }));
    }
    if (algorithm === 'sarsa') {
      const qHistory = sarsa(config, alpha, epsilon, episodes);
      return qHistory.map((q) => {
        const policy = epsilonGreedyPolicy(q, 0); // show greedy policy
        const values = actionValueToStateValue(q);
        return { values, policy };
      });
    }
    const qHistory = qLearning(config, alpha, epsilon, episodes);
    return qHistory.map((q) => {
      const policy = epsilonGreedyPolicy(q, 0);
      const values = actionValueToStateValue(q);
      return { values, policy };
    });
  }, [algorithm, alpha, epsilon, episodes, config]);

  const current = history[Math.min(step, history.length - 1)];
  const maxStep = history.length - 1;

  function next() {
    setStep((s) => Math.min(s + 1, maxStep));
  }
  function prev() {
    setStep((s) => Math.max(s - 1, 0));
  }
  function reset() {
    setStep(0);
  }
  function selectAlgorithm(a: Algorithm) {
    setAlgorithm(a);
    setStep(0);
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Clock className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          7.1 时序差分方法
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          在 GridWorld 上对比 TD(0)、Sarsa 与 Q-learning 的值函数与策略演化。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="TD(0) 更新"
          formula={<KaTeX math={String.raw`v(s_t) \leftarrow v(s_t) + \alpha \bigl[ r_{t+1} + \gamma v(s_{t+1}) - v(s_t) \bigr]`} display />}
          description="每走一步就更新当前状态的值，目标中使用了下一个状态的估计值（自举）。"
        />
        <FormulaCard
          title="Sarsa 更新"
          formula={<KaTeX math={String.raw`q(s_t,a_t) \leftarrow q(s_t,a_t) + \alpha \bigl[ r_{t+1} + \gamma q(s_{t+1},a_{t+1}) - q(s_t,a_t) \bigr]`} display />}
          description="同策略：用实际执行的下一个动作 a_{t+1} 构造 TD 目标。"
        />
        <FormulaCard
          title="Q-learning 更新"
          formula={<KaTeX math={String.raw`q(s_t,a_t) \leftarrow q(s_t,a_t) + \alpha \bigl[ r_{t+1} + \gamma \max_a q(s_{t+1},a) - q(s_t,a_t) \bigr]`} display />}
          description="异策略：用下一个状态的最大动作值构造 TD 目标，而不管实际执行哪个动作。"
        />
      </section>

      <InteractiveDemo title="TD 算法逐回合演化">
        <div className="grid lg:grid-cols-[1fr_340px] gap-6">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld
              config={config}
              policy={current.policy}
              values={current.values}
              showValues
              className="max-w-full"
            />
            <p className="mt-4 text-sm text-gray-500 text-center">
              第 {step} 回合后的{algorithm === 'td0' ? 'TD(0) 状态值' : algorithm === 'sarsa' ? 'Sarsa 贪心策略' : 'Q-learning 贪心策略'}
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">选择算法</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-2">
                <Button
                  variant={algorithm === 'td0' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => selectAlgorithm('td0')}
                >
                  TD(0)
                </Button>
                <Button
                  variant={algorithm === 'sarsa' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => selectAlgorithm('sarsa')}
                >
                  Sarsa
                </Button>
                <Button
                  variant={algorithm === 'qlearning' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => selectAlgorithm('qlearning')}
                >
                  Q-learning
                </Button>
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
                {algorithm !== 'td0' && (
                  <div>
                    <div className="text-sm text-gray-600 mb-1">探索率 ε</div>
                    <Slider value={[epsilon]} min={0} max={1} step={0.05} onValueChange={([v]) => setEpsilon(v)} />
                    <div className="mt-1 text-center font-mono text-sm text-gray-700">{epsilon.toFixed(2)}</div>
                  </div>
                )}
                <div>
                  <div className="text-sm text-gray-600 mb-1">训练回合数</div>
                  <Slider value={[episodes]} min={10} max={300} step={10} onValueChange={([v]) => setEpisodes(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{episodes}</div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">回放控制</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-sm text-gray-600">
                  <span>回合</span>
                  <span className="font-mono font-semibold">{step} / {maxStep}</span>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={prev} disabled={step === 0} className="flex-1">上一步</Button>
                  <Button size="sm" onClick={next} disabled={step === maxStep} className="flex-1">下一步</Button>
                </div>
                <Button variant="outline" size="sm" onClick={reset} className="w-full">重置</Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </InteractiveDemo>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">观察与思考</h2>
        <ul className="space-y-3 text-gray-700">
          <li><strong>TD vs MC：</strong>TD 每步更新，不需要等回合结束；MC 需要完整回合。</li>
          <li><strong>Sarsa vs Q-learning：</strong>Sarsa 更保守（同策略），Q-learning 更激进（异策略），对探索的敏感度不同。</li>
          <li><strong>学习率：</strong>α 过大时 TD 可能震荡；α 过小则收敛缓慢。</li>
        </ul>
      </section>
    </div>
  );
}
