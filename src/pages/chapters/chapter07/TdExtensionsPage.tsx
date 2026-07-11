import { useState, useMemo, useEffect } from 'react';
import { FlaskConical, ShieldAlert } from 'lucide-react';
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
  greedyPolicy,
  expectedSarsa,
  sarsaLambda,
  tdLambdaPrediction,
  actionValueToStateValue,
  policyBellmanResidualV,
  policyBellmanResidualQ,
  type Action,
  type Policy,
  type PredictionFrame,
  type ControlFrame,
} from '@/lib/rl/gridworld';

type ExtAlgorithm = 'expected' | 'sarsa-lambda' | 'td-lambda';

const RIGHT_POLICY: Action[] = [1, 1, 1, 1, 1, 1, 1, 1, 1];
const HORIZON = 30;
const SEED = 1;

function frameToView(frame: PredictionFrame | ControlFrame): {
  values: number[];
  policy: Policy;
} {
  if (frame.kind === 'v') {
    return { values: frame.values, policy: frame.policy };
  }
  return {
    values: actionValueToStateValue(frame.qValues),
    policy: greedyPolicy(frame.qValues),
  };
}

export default function Chapter07TdExtensionsPage() {
  const [algorithm, setAlgorithm] = useState<ExtAlgorithm>('expected');
  const [alpha, setAlpha] = useState(0.2);
  const [epsilon, setEpsilon] = useState(0.3);
  const [lambda, setLambda] = useState(0.8);
  const [episodes, setEpisodes] = useState(100);
  const [step, setStep] = useState(0);

  const config = DEFAULT_CONFIG;

  useEffect(() => {
    setStep(0);
  }, [algorithm, alpha, epsilon, lambda, episodes]);

  const result = useMemo(() => {
    if (algorithm === 'td-lambda') {
      const policy = deterministicPolicy(RIGHT_POLICY, 5);
      return tdLambdaPrediction(policy, config, alpha, lambda, HORIZON, episodes, SEED);
    }
    if (algorithm === 'sarsa-lambda') {
      return sarsaLambda(config, alpha, epsilon, 'fixed', lambda, HORIZON, episodes, SEED);
    }
    return expectedSarsa(config, alpha, epsilon, 'fixed', HORIZON, episodes, SEED);
  }, [algorithm, alpha, epsilon, lambda, episodes, config]);

  const frames = result.frames;
  const currentFrame = frames[Math.min(step, frames.length - 1)];
  const currentView = frameToView(currentFrame);
  const maxStep = frames.length - 1;

  const convergenceData = useMemo(() => {
    const expectedHist = expectedSarsa(config, alpha, epsilon, 'fixed', HORIZON, episodes, SEED).frames.map(
      (f) => actionValueToStateValue(f.qValues)[0]
    );
    const sarsaLambdaHist = sarsaLambda(config, alpha, epsilon, 'fixed', lambda, HORIZON, episodes, SEED).frames.map(
      (f) => actionValueToStateValue(f.qValues)[0]
    );
    const tdLambdaHist = tdLambdaPrediction(
      deterministicPolicy(RIGHT_POLICY, 5),
      config,
      alpha,
      lambda,
      HORIZON,
      episodes,
      SEED
    ).frames.map((f) => f.values[0]);
    const maxLen = Math.max(expectedHist.length, sarsaLambdaHist.length, tdLambdaHist.length);
    return Array.from({ length: maxLen }, (_, i) => ({
      episode: i,
      expected: expectedHist[i] ?? null,
      sarsaLambda: sarsaLambdaHist[i] ?? null,
      tdLambda: tdLambdaHist[i] ?? null,
    }));
  }, [config, alpha, epsilon, lambda, episodes]);

  const tdErrorData = useMemo(() => {
    return frames.slice(1).map((frame, i) => {
      let residual = 0;
      if (frame.kind === 'v') {
        residual = policyBellmanResidualV(frame.values, frame.policy, config);
      } else {
        residual = policyBellmanResidualQ(frame.qValues, frame.behaviorPolicy, config);
      }
      return {
        episode: i + 1,
        residual,
      };
    });
  }, [frames, config, algorithm]);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-purple-100 rounded-xl flex items-center justify-center">
            <FlaskConical className="w-8 h-8 text-purple-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          第 7 章 教材拓展
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Expected Sarsa、TD(λ) 与 Sarsa(λ) 是教材之外的延伸内容，用于理解更稳定或更高效的 TD 变体。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
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
          title="TD(λ) 预测"
          formula={
            <KaTeX
              math={String.raw`\delta_t = r_{t+1} + \gamma v(s_{t+1}) - v(s_t), \quad E_t(s) = \gamma \lambda E_{t-1}(s) + \mathbf{1}_{s=s_t}`}
              display
            />
          }
          description="为每个状态维护资格迹，把当前 TD 误差按迹分配给近期访问过的状态。"
        />
        <FormulaCard
          title="Sarsa(λ) 控制"
          formula={
            <KaTeX
              math={String.raw`E_t(s,a) = \gamma \lambda E_{t-1}(s,a) + \mathbf{1}_{s=s_t,a=a_t}, \quad q \leftarrow q + \alpha \delta_t E_t`}
              display
            />
          }
          description="将资格迹从状态值推广到动作值，是同策略 TD 控制的多步扩展。"
        />
      </section>

      <InteractiveDemo title="拓展算法逐回合演化">
        <div className="grid lg:grid-cols-[1fr_340px] gap-6">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld
              config={config}
              policy={currentView.policy}
              values={currentView.values}
              showValues
              className="max-w-full"
            />
            <p className="mt-4 text-sm text-gray-500 text-center">
              第 {step} 回合后的{displayName(algorithm)}
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">选择算法（教材拓展）</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-1 gap-2">
                {[
                  { key: 'expected', label: 'Expected Sarsa' },
                  { key: 'sarsa-lambda', label: 'Sarsa(λ)' },
                  { key: 'td-lambda', label: 'TD(λ)' },
                ].map(({ key, label }) => (
                  <Button
                    key={key}
                    variant={algorithm === key ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setAlgorithm(key as ExtAlgorithm)}
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
                {(algorithm === 'expected' || algorithm === 'sarsa-lambda') && (
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
            { key: 'expected', name: 'Expected Sarsa', color: '#f59e0b' },
            { key: 'sarsaLambda', name: 'Sarsa(λ)', color: '#ef4444' },
            { key: 'tdLambda', name: 'TD(λ)', color: '#2563eb' },
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
          对 TD(λ) 计算策略 Bellman 残差；对 Expected Sarsa 与 Sarsa(λ) 计算同策略 Bellman 残差。
        </p>
      </InteractiveDemo>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">小结与 Q&A</h2>
        <ConceptAccordion
          items={[
            {
              id: 'summary',
              title: '本节小结',
              content: (
                <ul className="list-disc pl-5 space-y-2">
                  <li>Expected Sarsa 用期望代替采样，目标方差更小。</li>
                  <li>资格迹把多步回报的信息高效地分配到多个状态-动作对。</li>
                  <li>λ=0 时退化为单步方法；λ 越接近 1，越接近多步/Monte Carlo 行为。</li>
                </ul>
              ),
            },
            {
              id: 'qa1',
              title: 'Q: λ=1 时资格迹等价于 MC 吗？',
              content:
                'λ=1 时迹不衰减，更新权重与 Monte Carlo 回报高度相关；但在线更新顺序与完整回报仍有差异，因此不应视为完全等价。',
            },
          ]}
        />
      </section>
    </div>
  );
}

function displayName(algorithm: ExtAlgorithm): string {
  switch (algorithm) {
    case 'expected':
      return 'Expected Sarsa 贪心策略';
    case 'sarsa-lambda':
      return 'Sarsa(λ) 贪心策略';
    case 'td-lambda':
      return 'TD(λ) 状态值';
  }
}
