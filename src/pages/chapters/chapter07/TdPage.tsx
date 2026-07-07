import { useState, useMemo, useEffect } from 'react';
import { Clock, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import AlgorithmPlayer from '@/components/AlgorithmPlayer';
import LineChart from '@/components/LineChart';
import ConceptAccordion from '@/components/ConceptAccordion';
import {
  DEFAULT_CONFIG,
  deterministicPolicy,
  step,
  tdZeroPrediction,
  qLearning,
  sarsa,
  expectedSarsa,
  nStepSarsa,
  sarsaLambda,
  tdLambdaPrediction,
  epsilonGreedyPolicy,
  actionValueToStateValue,
  type Action,
} from '@/lib/rl/gridworld';

type Algorithm =
  | 'td0'
  | 'sarsa'
  | 'expected'
  | 'nstep'
  | 'qlearning'
  | 'sarsa-lambda'
  | 'td-lambda';

const RIGHT_POLICY: Action[] = [1, 1, 1, 1, 1, 1, 1, 1, 1];

export default function Chapter07TdPage() {
  const [algorithm, setAlgorithm] = useState<Algorithm>('td0');
  const [alpha, setAlpha] = useState(0.2);
  const [epsilon, setEpsilon] = useState(0.3);
  const [episodes, setEpisodes] = useState(100);
  const [nStep, setNStep] = useState(3);
  const [lambda, setLambda] = useState(0.8);
  const [step, setStep] = useState(0);

  const config = DEFAULT_CONFIG;

  useEffect(() => {
    setStep(0);
  }, [algorithm, alpha, epsilon, episodes, nStep, lambda]);

  const history = useMemo(() => {
    if (algorithm === 'td0') {
      const policy = deterministicPolicy(RIGHT_POLICY, 5);
      const values = tdZeroPrediction(policy, config, alpha, episodes);
      return values.map((v) => ({ values: v, policy }));
    }
    if (algorithm === 'td-lambda') {
      const policy = deterministicPolicy(RIGHT_POLICY, 5);
      const values = tdLambdaPrediction(policy, config, alpha, lambda, episodes);
      return values.map((v) => ({ values: v, policy }));
    }
    if (algorithm === 'sarsa') {
      const qHistory = sarsa(config, alpha, epsilon, episodes);
      return qHistory.map((q) => {
        const policy = epsilonGreedyPolicy(q, 0);
        const values = actionValueToStateValue(q);
        return { values, policy };
      });
    }
    if (algorithm === 'expected') {
      const qHistory = expectedSarsa(config, alpha, epsilon, episodes);
      return qHistory.map((q) => {
        const policy = epsilonGreedyPolicy(q, 0);
        const values = actionValueToStateValue(q);
        return { values, policy };
      });
    }
    if (algorithm === 'nstep') {
      const qHistory = nStepSarsa(config, alpha, epsilon, nStep, episodes);
      return qHistory.map((q) => {
        const policy = epsilonGreedyPolicy(q, 0);
        const values = actionValueToStateValue(q);
        return { values, policy };
      });
    }
    if (algorithm === 'sarsa-lambda') {
      const qHistory = sarsaLambda(config, alpha, epsilon, lambda, episodes);
      return qHistory.map((q) => {
        const policy = epsilonGreedyPolicy(q, 0);
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
  }, [algorithm, alpha, epsilon, episodes, nStep, lambda, config]);

  const current = history[Math.min(step, history.length - 1)];
  const maxStep = history.length - 1;

  const convergenceData = useMemo(() => {
    const policy = deterministicPolicy(RIGHT_POLICY, 5);
    const td0 = tdZeroPrediction(policy, config, alpha, episodes).map((v) => v[0]);
    const sarsaHist = sarsa(config, alpha, epsilon, episodes).map((q) => actionValueToStateValue(q)[0]);
    const expectedHist = expectedSarsa(config, alpha, epsilon, episodes).map((q) => actionValueToStateValue(q)[0]);
    const qHist = qLearning(config, alpha, epsilon, episodes).map((q) => actionValueToStateValue(q)[0]);
    const maxLen = Math.max(td0.length, sarsaHist.length, expectedHist.length, qHist.length);
    return Array.from({ length: maxLen }, (_, i) => ({
      episode: i,
      td0: td0[i] ?? null,
      sarsa: sarsaHist[i] ?? null,
      expected: expectedHist[i] ?? null,
      qlearning: qHist[i] ?? null,
    }));
  }, [config, alpha, epsilon, episodes]);

  const tdErrorData = useMemo(() => {
    return history.slice(1).map((frame, i) => ({
      episode: i + 1,
      residual: computeResidual(frame.values, frame.policy, config, algorithm),
    }));
  }, [history, config, algorithm]);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Clock className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          第 7 章 时序差分方法
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          TD(0)、Sarsa、Expected Sarsa、n-step Sarsa、Q-learning 以及带资格迹的 TD(λ) / Sarsa(λ)。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="TD(0) 预测"
          formula={<KaTeX math={String.raw`v(s_t) \leftarrow v(s_t) + \alpha \bigl[ r_{t+1} + \gamma v(s_{t+1}) - v(s_t) \bigr]`} display />}
          description="每接收一个转移样本就更新一次状态值，是最简单的自举方法。"
        />
        <FormulaCard
          title="Sarsa（同策略控制）"
          formula={<KaTeX math={String.raw`q(s_t,a_t) \leftarrow q(s_t,a_t) + \alpha \bigl[ r_{t+1} + \gamma q(s_{t+1},a_{t+1}) - q(s_t,a_t) \bigr]`} display />}
          description="TD 目标中的下一个动作 a_{t+1} 来自当前策略实际采样。"
        />
        <FormulaCard
          title="Expected Sarsa"
          formula={
            <KaTeX
              math={String.raw`q(s_t,a_t) \leftarrow q(s_t,a_t) + \alpha \bigl[ r_{t+1} + \gamma \sum_a \pi(a|s_{t+1}) q(s_{t+1},a) - q(s_t,a_t) \bigr]`}
              display
            />
          }
          description="TD 目标使用下一状态的动作期望，通常比 Sarsa 更稳定。"
        />
        <FormulaCard
          title="n-step Sarsa"
          formula={
            <KaTeX
              math={String.raw`G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n q(s_{t+n}, a_{t+n})`}
              display
            />
          }
          description="在 TD（n=1）和 MC（n=∞）之间做 bias-variance 权衡。"
        />
        <FormulaCard
          title="Q-learning（异策略控制）"
          formula={
            <KaTeX
              math={String.raw`q(s_t,a_t) \leftarrow q(s_t,a_t) + \alpha \bigl[ r_{t+1} + \gamma \max_a q(s_{t+1},a) - q(s_t,a_t) \bigr]`}
              display
            />
          }
          description="TD 目标使用下一个状态的最大动作值，即使实际执行的动作不是它。"
        />
        <FormulaCard
          title="资格迹"
          formula={
            <KaTeX
              math={String.raw`E_t(s,a) = \gamma \lambda E_{t-1}(s,a) + \mathbf{1}_{s=s_t,a=a_t}, \quad q \leftarrow q + \alpha \delta_t E_t`}
              display
            />
          }
          description="λ=0 退化为单步 TD；λ=1 接近蒙特卡洛。资格迹把近期访问过的 (s,a) 与当前 TD 误差联系起来。"
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
              第 {step} 回合后的{displayName(algorithm, nStep)}
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">选择算法</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-2">
                {[
                  { key: 'td0', label: 'TD(0)' },
                  { key: 'td-lambda', label: 'TD(λ)' },
                  { key: 'sarsa', label: 'Sarsa' },
                  { key: 'expected', label: 'Expected Sarsa' },
                  { key: 'nstep', label: 'n-step' },
                  { key: 'sarsa-lambda', label: 'Sarsa(λ)' },
                  { key: 'qlearning', label: 'Q-learning' },
                ].map(({ key, label }) => (
                  <Button
                    key={key}
                    variant={algorithm === key ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setAlgorithm(key as Algorithm)}
                  >
                    {label}
                  </Button>
                ))}
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
                {(algorithm === 'sarsa' || algorithm === 'expected' || algorithm === 'nstep' || algorithm === 'sarsa-lambda' || algorithm === 'qlearning') && (
                  <div>
                    <div className="text-sm text-gray-600 mb-1">探索率 ε</div>
                    <Slider value={[epsilon]} min={0} max={1} step={0.05} onValueChange={([v]) => setEpsilon(v)} />
                    <div className="mt-1 text-center font-mono text-sm text-gray-700">{epsilon.toFixed(2)}</div>
                  </div>
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
              <CardContent>
                <AlgorithmPlayer maxStep={maxStep} currentStep={step} onStepChange={setStep} />
              </CardContent>
            </Card>
          </div>
        </div>
      </InteractiveDemo>

      <InteractiveDemo title="状态值收敛对比">
        <LineChart
          data={convergenceData}
          xKey="episode"
          xLabel="回合"
          yLabel="v(s_1)"
          series={[
            { key: 'td0', name: 'TD(0)', color: '#2563eb' },
            { key: 'sarsa', name: 'Sarsa', color: '#ef4444' },
            { key: 'expected', name: 'Expected Sarsa', color: '#f59e0b' },
            { key: 'qlearning', name: 'Q-learning', color: '#22c55e' },
          ]}
        />
      </InteractiveDemo>

      <InteractiveDemo title="单步 TD 误差 / Bellman 残差">
        <LineChart
          data={tdErrorData}
          xKey="episode"
          xLabel="回合"
          yLabel="max |δ|"
          series={[{ key: 'residual', name: '最大单步误差', color: '#8b5cf6' }]}
        />
        <p className="mt-2 text-sm text-gray-600">
          对预测算法计算策略 Bellman 残差；对控制算法计算最优 Bellman 残差。随着训练进行，该值应逐渐下降。
        </p>
      </InteractiveDemo>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">小结与 Q&A</h2>
        <ConceptAccordion
          items={[
            {
              id: 'summary',
              title: '本章小结',
              content: (
                <ul className="list-disc pl-5 space-y-2">
                  <li>TD 算法是随机逼近求解贝尔曼/贝尔曼最优方程的特例。</li>
                  <li>TD(0) 每步更新，是在线学习的典型代表。</li>
                  <li>Sarsa 同策略、Q-learning 异策略，Expected Sarsa 通常更稳定。</li>
                  <li>n-step 方法在 TD 和 MC 之间做 bias-variance 权衡。</li>
                  <li>资格迹（eligibility traces）把多步回报的高效计算与单步更新结合起来。</li>
                </ul>
              ),
            },
            {
              id: 'qa1',
              title: 'Q: TD 与 MC 的核心区别是什么？',
              content:
                'MC 需要等一个完整回合后才能更新；TD 每走一步就可以用自举目标更新，因此可以在线学习。',
            },
            {
              id: 'qa2',
              title: 'Q: λ=0 和 λ=1 分别对应什么？',
              content: 'λ=0 时资格迹只保留当前步，退化为单步 TD/Sarsa；λ=1 时资格迹不衰减，更新等价于完整的 Monte Carlo 回报。',
            },
          ]}
        />
      </section>
    </div>
  );
}

function displayName(algorithm: Algorithm, nStep: number): string {
  switch (algorithm) {
    case 'td0':
      return 'TD(0) 状态值';
    case 'td-lambda':
      return 'TD(λ) 状态值';
    case 'sarsa':
      return 'Sarsa 贪心策略';
    case 'expected':
      return 'Expected Sarsa 贪心策略';
    case 'nstep':
      return `${nStep}-step Sarsa 贪心策略`;
    case 'sarsa-lambda':
      return 'Sarsa(λ) 贪心策略';
    case 'qlearning':
      return 'Q-learning 贪心策略';
  }
}

function computeResidual(
  values: number[],
  policy: number[][],
  config: typeof DEFAULT_CONFIG,
  algorithm: Algorithm
): number {
  const numStates = config.rows * config.cols;

  // Prediction algorithms: policy Bellman residual
  if (algorithm === 'td0' || algorithm === 'td-lambda') {
    let maxResidual = 0;
    for (let s = 0; s < numStates; s++) {
      // deterministic policy: take the action with probability 1
      const action = policy[s].findIndex((p) => p > 0.5) as Action;
      const result = step(s, action, config);
      const delta = result.reward + config.gamma * values[result.nextState] - values[s];
      maxResidual = Math.max(maxResidual, Math.abs(delta));
    }
    return maxResidual;
  }

  // Control algorithms: optimal Bellman residual using current greedy values
  const q = values.map((_, s) => {
    return Array.from({ length: 5 }, (_, a) => {
      const result = step(s, a as Action, config);
      return result.reward + config.gamma * values[result.nextState];
    });
  });
  let maxResidual = 0;
  for (let s = 0; s < numStates; s++) {
    for (let a = 0; a < 5; a++) {
      const result = step(s, a as Action, config);
      const maxQNext = Math.max(...q[result.nextState]);
      const target = result.reward + config.gamma * maxQNext;
      const delta = target - q[s][a];
      maxResidual = Math.max(maxResidual, Math.abs(delta));
    }
  }
  return maxResidual;
}
