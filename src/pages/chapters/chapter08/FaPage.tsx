import { useState, useMemo } from 'react';
import { Brain, ShieldAlert } from 'lucide-react';
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
import ConceptAccordion from '@/components/ConceptAccordion';
import {
  DEFAULT_CONFIG,
  ACTION_NAMES,
  deterministicPolicy,
  greedyPolicy,
  actionValueToStateValue,
  estimateTrueActionValues,
  qTableRMSE,
} from '@/lib/rl/gridworld';
import {
  semiGradientTD,
  dqnGridWorld,
  actionValueFA,
  type FeatureMode,
  type ActionValueFeatureMode,
} from '@/lib/rl/fa';

type TabKey = 'polynomial' | 'semi-gradient' | 'action-value' | 'dqn';

const RIGHT_POLICY: (0 | 1 | 2 | 3 | 4)[] = [1, 1, 1, 1, 1, 1, 1, 1, 1];

export default function Chapter08FaPage() {
  const [activeTab, setActiveTab] = useState<TabKey>('polynomial');

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Brain className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          第 8 章 值函数近似
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          从线性函数近似到深度 Q 网络：用参数化函数逼近值函数，并通过经验回放与目标网络稳定学习。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="线性值函数近似"
          formula={<KaTeX math={String.raw`\hat{v}(s, w) = \phi(s)^\top w = \sum_{i=0}^d \phi_i(s) w_i`} display />}
          description="用状态特征 φ(s) 的线性组合来近似状态值，参数 w 决定了近似形状。"
        />
        <FormulaCard
          title="半梯度 TD 更新"
          formula={
            <KaTeX
              math={String.raw`w \leftarrow w + \alpha \bigl[ r + \gamma \hat{v}(s', w) - \hat{v}(s, w) \bigr] \nabla_w \hat{v}(s, w)`}
              display
            />
          }
          description="只对目标中的样本梯度进行更新，因此称为半梯度（semi-gradient）方法。"
        />
        <FormulaCard
          title="DQN 核心机制"
          formula={
            <KaTeX
              math={String.raw`L(w) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \bigl[ (r + \gamma \max_{a'} Q(s', a'; w^-) - Q(s,a; w))^2 \bigr]`}
              display
            />
          }
          description="经验回放 D 打破样本相关性，目标网络 w^- 稳定 bootstrap 目标。"
        />
      </section>

      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as TabKey)}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="polynomial">多项式拟合</TabsTrigger>
          <TabsTrigger value="semi-gradient">半梯度 TD(λ)</TabsTrigger>
          <TabsTrigger value="action-value">动作值近似 q(s,a,w)</TabsTrigger>
          <TabsTrigger value="dqn">Deep Q-learning / DQN</TabsTrigger>
        </TabsList>

        <TabsContent value="polynomial" className="mt-4">
          <PolynomialFittingDemo />
        </TabsContent>
        <TabsContent value="semi-gradient" className="mt-4">
          <SemiGradientDemo />
        </TabsContent>
        <TabsContent value="action-value" className="mt-4">
          <ActionValueFADemo />
        </TabsContent>
        <TabsContent value="dqn" className="mt-4">
          <DQNDemo />
        </TabsContent>
      </Tabs>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">小结与 Q&A</h2>
        <ConceptAccordion
          items={[
            {
              id: 'summary',
              title: '本章小结',
              content: (
                <ul className="list-disc pl-5 space-y-2">
                  <li>函数近似用有限参数逼近大状态空间中的值函数。</li>
                  <li>线性近似中，one-hot 特征等价于表格法；坐标/多项式特征可实现泛化。</li>
                  <li>半梯度 TD 使用自举目标，更新不是真正的随机梯度下降，但在 mild 条件下收敛。</li>
                  <li>DQN 用神经网络做非线性近似，并通过经验回放和目标网络稳定训练。</li>
                </ul>
              ),
            },
            {
              id: 'qa1',
              title: 'Q: 为什么叫“半梯度”？',
              content:
                '因为 TD 目标 r + γv̂(s\',w) 本身也依赖于参数 w，但我们只把目标当作常数，对预测值 v̂(s,w) 求梯度，所以只走了“半个”梯度步。',
            },
            {
              id: 'qa2',
              title: 'Q: 目标网络为什么能稳定 DQN？',
              content:
                '如果用一个网络同时计算当前 Q 值和 bootstrap 目标，目标会随每次更新而变化，导致训练目标像“追着自己的尾巴跑”。目标网络定期复制主网络参数，使目标在若干步内保持相对稳定。',
            },
          ]}
        />
      </section>
    </div>
  );
}

// ------------------- Polynomial fitting (original demo) -------------------
function polynomialFeatures(x: number, degree: number): number[] {
  const phi = [1];
  for (let i = 1; i <= degree; i++) {
    phi.push(phi[i - 1] * x);
  }
  return phi;
}

function trueValue(x: number): number {
  return Math.sin(Math.PI * x);
}

function PolynomialFittingDemo() {
  const [degree, setDegree] = useState(3);
  const [alpha, setAlpha] = useState(0.05);
  const [iterations, setIterations] = useState(200);
  const [noiseStd, setNoiseStd] = useState(0.1);
  const [seed, setSeed] = useState(0);

  const { weights, predictions } = useMemo(() => {
    void seed;
    const numFeatures = degree + 1;
    let w = new Array(numFeatures).fill(0);
    const n = 50;

    for (let k = 0; k < iterations; k++) {
      const x = Math.random() * 2 - 1;
      const phi = polynomialFeatures(x, degree);
      const target = trueValue(x) + (Math.random() * 2 - 1) * noiseStd;
      const pred = phi.reduce((sum, f, i) => sum + f * w[i], 0);
      const error = target - pred;
      for (let i = 0; i < numFeatures; i++) {
        w[i] += alpha * error * phi[i];
      }
    }

    const xs = Array.from({ length: n }, (_, i) => -1 + (2 * i) / (n - 1));
    const preds = xs.map((x) => {
      const phi = polynomialFeatures(x, degree);
      return phi.reduce((sum, f, i) => sum + f * w[i], 0);
    });

    return { weights: w, predictions: { xs, preds } };
  }, [degree, alpha, iterations, noiseStd, seed]);

  const width = 360;
  const height = 220;
  const padding = 28;
  const plotWidth = width - 2 * padding;
  const plotHeight = height - 2 * padding;
  const minY = -1.5;
  const maxY = 1.5;

  function scaleX(x: number) {
    return padding + ((x + 1) / 2) * plotWidth;
  }
  function scaleY(y: number) {
    return padding + (1 - (y - minY) / (maxY - minY)) * plotHeight;
  }

  const truePathD = predictions.xs
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(x)} ${scaleY(trueValue(x))}`)
    .join(' ');

  const predPathD = predictions.xs
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(x)} ${scaleY(predictions.preds[i])}`)
    .join(' ');

  return (
    <InteractiveDemo title="多项式逼近状态值函数">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
          <svg width={width} height={height} className="bg-white rounded-lg border border-gray-200">
            <line x1={padding} y1={scaleY(0)} x2={width - padding} y2={scaleY(0)} stroke="#e5e7eb" strokeWidth={1} />
            <path d={truePathD} fill="none" stroke="#22c55e" strokeWidth={2} />
            <path d={predPathD} fill="none" stroke="#2563eb" strokeWidth={2} strokeDasharray="4 2" />
            <text x={width - padding} y={scaleY(1.3)} textAnchor="end" fontSize={10} fill="#22c55e">
              真实值 v*(s)
            </text>
            <text x={width - padding} y={scaleY(1.0)} textAnchor="end" fontSize={10} fill="#2563eb">
              近似值 v̂(s,w)
            </text>
            <text x={padding} y={height - 4} fontSize={10} fill="#6b7280">状态 s</text>
            <text x={4} y={padding - 4} fontSize={10} fill="#6b7280">v</text>
          </svg>
          <p className="mt-4 text-sm text-gray-500 text-center">
            绿色实线为真实值函数，蓝色虚线为线性近似结果
          </p>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">模型设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="text-sm text-gray-600 mb-1">多项式阶数 d</div>
                <Slider value={[degree]} min={0} max={8} step={1} onValueChange={([v]) => setDegree(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{degree}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">学习率 α</div>
                <Slider value={[alpha]} min={0.001} max={0.2} step={0.001} onValueChange={([v]) => setAlpha(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{alpha.toFixed(3)}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">训练迭代次数</div>
                <Slider value={[iterations]} min={0} max={1000} step={50} onValueChange={([v]) => setIterations(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{iterations}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">目标噪声</div>
                <Slider value={[noiseStd]} min={0} max={0.5} step={0.05} onValueChange={([v]) => setNoiseStd(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{noiseStd.toFixed(2)}</div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">参数向量 w</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="font-mono text-sm text-gray-700 break-all">
                [{weights.map((wi) => wi.toFixed(2)).join(', ')}]
              </div>
            </CardContent>
          </Card>

          <Button onClick={() => setSeed((s) => s + 1)} variant="outline" className="w-full">
            重新采样训练
          </Button>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- Semi-gradient TD(lambda) -------------------
function SemiGradientDemo() {
  const config = DEFAULT_CONFIG;
  const policy = useMemo(() => deterministicPolicy(RIGHT_POLICY, 5), []);
  const [featureMode, setFeatureMode] = useState<FeatureMode>('coordinate');
  const [degree, setDegree] = useState(2);
  const [alpha, setAlpha] = useState(0.05);
  const [lambda, setLambda] = useState(0.8);
  const [episodes, setEpisodes] = useState(200);

  const result = useMemo(() => {
    return semiGradientTD(policy, config, {
      alpha,
      lambda,
      featureMode,
      polynomialDegree: degree,
      episodes,
      maxSteps: 30,
    });
  }, [policy, config, alpha, lambda, featureMode, degree, episodes]);

  const rmseData = useMemo(() => {
    return result.valuesHistory.map((v, i) => ({
      episode: i,
      rmse: Math.sqrt(
        v.reduce((sum, val, s) => sum + Math.pow(val - result.trueValues[s], 2), 0) / v.length
      ),
    }));
  }, [result]);

  const finalValues = result.valuesHistory[result.valuesHistory.length - 1];
  const finalRMSE = rmseData[rmseData.length - 1]?.rmse ?? 0;

  return (
    <InteractiveDemo title="半梯度 TD(λ)：线性函数近似策略评估">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="grid md:grid-cols-2 gap-4">
            <div className="flex flex-col items-center bg-gray-50 rounded-xl p-4 border border-gray-200">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">预测值 v̂(s,w)</h3>
              <GridWorld config={config} values={finalValues} showValues className="max-w-full" />
            </div>
            <div className="flex flex-col items-center bg-gray-50 rounded-xl p-4 border border-gray-200">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">真实值 v_π(s)</h3>
              <GridWorld config={config} values={result.trueValues} showValues className="max-w-full" />
            </div>
          </div>
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <LineChart
              data={rmseData}
              xKey="episode"
              xLabel="回合"
              yLabel="RMSE(v̂, v_π)"
              series={[{ key: 'rmse', name: '值近似 RMSE', color: '#2563eb' }]}
              height={220}
            />
          </div>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm text-gray-700 block mb-1">特征构造</label>
                <Select value={featureMode} onValueChange={(v) => setFeatureMode(v as FeatureMode)}>
                  <SelectTrigger>
                    <SelectValue placeholder="选择特征" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="onehot">One-hot（表格等价）</SelectItem>
                    <SelectItem value="coordinate">坐标归一化</SelectItem>
                    <SelectItem value="polynomial">坐标多项式</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {featureMode === 'polynomial' && (
                <div>
                  <div className="text-sm text-gray-600 mb-1">多项式阶数</div>
                  <Slider value={[degree]} min={1} max={4} step={1} onValueChange={([v]) => setDegree(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{degree}</div>
                </div>
              )}
              <div>
                <div className="text-sm text-gray-600 mb-1">学习率 α</div>
                <Slider value={[alpha]} min={0.001} max={0.2} step={0.001} onValueChange={([v]) => setAlpha(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{alpha.toFixed(3)}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">资格迹衰减 λ</div>
                <Slider value={[lambda]} min={0} max={0.99} step={0.01} onValueChange={([v]) => setLambda(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{lambda.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">训练回合数</div>
                <Slider value={[episodes]} min={10} max={500} step={10} onValueChange={([v]) => setEpisodes(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{episodes}</div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">结果</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <div>最终 RMSE：<span className="font-mono font-semibold">{finalRMSE.toFixed(4)}</span></div>
              <div>权重维度：<span className="font-mono">{result.weightsHistory[0].length}</span></div>
              <div className="font-mono text-xs break-all">
                w ≈ [{result.weightsHistory[result.weightsHistory.length - 1].map((w) => w.toFixed(2)).join(', ')}]
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- DQN skeleton -------------------
function ActionValueFADemo() {
  const config = DEFAULT_CONFIG;
  const qStar = useMemo(() => estimateTrueActionValues(config), [config]);
  const [algorithm, setAlgorithm] = useState<'sarsa' | 'qlearning'>('sarsa');
  const [featureMode, setFeatureMode] = useState<ActionValueFeatureMode>('shared');
  const [alpha, setAlpha] = useState(0.05);
  const [epsilon, setEpsilon] = useState(0.3);
  const [episodes, setEpisodes] = useState(200);
  const [seed, setSeed] = useState(0);

  const result = useMemo(() => {
    void seed;
    return actionValueFA(config, {
      alpha,
      epsilon,
      gamma: config.gamma,
      episodes,
      maxSteps: 30,
      featureMode,
      algorithm,
    });
  }, [config, algorithm, featureMode, alpha, epsilon, episodes, seed]);

  const finalQ = result.qHistory[result.qHistory.length - 1];
  const finalPolicy = useMemo(() => greedyPolicy(finalQ), [finalQ]);
  const finalValues = useMemo(() => actionValueToStateValue(finalQ), [finalQ]);
  const finalRMSE = useMemo(() => qTableRMSE(finalQ, qStar), [finalQ, qStar]);

  const last = result.lastUpdate;

  return (
    <InteractiveDemo title="动作值函数近似 q(s,a,w)">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld config={config} policy={finalPolicy} values={finalValues} showValues className="max-w-full" />
            <p className="mt-3 text-sm text-gray-500 text-center">训练结束后的贪心策略与状态值</p>
          </div>
          {last && (
            <div className="bg-white rounded-xl border border-gray-200 p-4 text-sm">
              <h3 className="font-semibold text-gray-800 mb-2">最近一次更新</h3>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-gray-700">
                <div>状态 s：<span className="font-mono">s{last.state + 1}</span></div>
                <div>动作 a：<span className="font-mono">{ACTION_NAMES[last.action]}</span></div>
                <div>奖励 r：<span className="font-mono">{last.reward.toFixed(2)}</span></div>
                <div>下一状态 s&apos;：<span className="font-mono">s{last.nextState + 1}</span></div>
                {last.nextAction !== undefined && (
                  <div>下一动作 a&apos;：<span className="font-mono">{ACTION_NAMES[last.nextAction]}</span></div>
                )}
                <div>预测 q(s,a,w)：<span className="font-mono">{last.prediction.toFixed(3)}</span></div>
                <div>目标 y：<span className="font-mono">{last.target.toFixed(3)}</span></div>
                <div>TD 误差 δ：<span className="font-mono">{last.tdError.toFixed(3)}</span></div>
                <div>权重变化 ‖Δw‖：<span className="font-mono">{last.weightChange.toFixed(5)}</span></div>
              </div>
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
                <label className="text-sm text-gray-700 block mb-1">算法</label>
                <Select value={algorithm} onValueChange={(v) => setAlgorithm(v as 'sarsa' | 'qlearning')}>
                  <SelectTrigger>
                    <SelectValue placeholder="选择算法" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="sarsa">Sarsa with FA</SelectItem>
                    <SelectItem value="qlearning">Q-learning with FA</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="text-sm text-gray-700 block mb-1">特征构造</label>
                <Select value={featureMode} onValueChange={(v) => setFeatureMode(v as ActionValueFeatureMode)}>
                  <SelectTrigger>
                    <SelectValue placeholder="选择特征" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="onehot">one-hot(s,a) — 表格等价</SelectItem>
                    <SelectItem value="shared">共享状态特征 + 动作 one-hot + 交互</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">学习率 α</div>
                <Slider value={[alpha]} min={0.001} max={0.2} step={0.001} onValueChange={([v]) => setAlpha(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{alpha.toFixed(3)}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">探索率 ε</div>
                <Slider value={[epsilon]} min={0} max={1} step={0.05} onValueChange={([v]) => setEpsilon(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{epsilon.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">训练回合数</div>
                <Slider value={[episodes]} min={50} max={500} step={50} onValueChange={([v]) => setEpisodes(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{episodes}</div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">结果</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-1">
              <div>最终 RMSE（相对 q*）：<span className="font-mono font-semibold">{finalRMSE.toFixed(4)}</span></div>
              <div>权重维度：<span className="font-mono">{result.weightsHistory[0].length}</span></div>
              <div className="text-xs text-gray-500 mt-1">
                共享特征包含：[1, rowNorm, colNorm, distanceToTarget, isForbidden] + action one-hot + 交互项
              </div>
            </CardContent>
          </Card>

          <Button onClick={() => setSeed((s) => s + 1)} variant="outline" className="w-full">
            重新随机训练
          </Button>
        </div>
      </div>
    </InteractiveDemo>
  );
}

function DQNDemo() {
  const config = DEFAULT_CONFIG;
  const qStar = useMemo(() => estimateTrueActionValues(config), [config]);
  const [hiddenSize, setHiddenSize] = useState(32);
  const [alpha, setAlpha] = useState(0.01);
  const [epsilon, setEpsilon] = useState(0.3);
  const [batchSize, setBatchSize] = useState(32);
  const [replayCapacity, setReplayCapacity] = useState(2000);
  const [targetUpdateInterval, setTargetUpdateInterval] = useState(100);
  const [episodes, setEpisodes] = useState(200);
  const [seed, setSeed] = useState(0);

  const result = useMemo(() => {
    void seed;
    return dqnGridWorld(config, {
      hiddenSize,
      alpha,
      epsilon,
      gamma: config.gamma,
      batchSize,
      replayCapacity,
      targetUpdateInterval,
      episodes,
      maxSteps: 30,
    });
  }, [config, hiddenSize, alpha, epsilon, batchSize, replayCapacity, targetUpdateInterval, episodes, seed]);

  const finalQ = result.qHistory[result.qHistory.length - 1];
  const finalPolicy = useMemo(() => greedyPolicy(finalQ), [finalQ]);
  const finalValues = useMemo(() => actionValueToStateValue(finalQ), [finalQ]);
  const finalRMSE = useMemo(() => qTableRMSE(finalQ, qStar), [finalQ, qStar]);

  const lossData = useMemo(() => {
    return result.lossHistory.map((loss, i) => ({
      step: i,
      loss,
    }));
  }, [result]);

  return (
    <InteractiveDemo title="DQN 骨架：神经网络 + 经验回放 + 目标网络">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld config={config} policy={finalPolicy} values={finalValues} showValues className="max-w-full" />
            <p className="mt-3 text-sm text-gray-500 text-center">训练结束后 DQN 的贪心策略与状态值</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <LineChart
              data={lossData}
              xKey="step"
              xLabel="训练步"
              yLabel="均方误差损失"
              series={[{ key: 'loss', name: 'DQN Loss', color: '#ef4444' }]}
              height={220}
            />
          </div>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">网络与训练设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="text-sm text-gray-600 mb-1">隐藏层大小</div>
                <Slider value={[hiddenSize]} min={8} max={64} step={8} onValueChange={([v]) => setHiddenSize(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{hiddenSize}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">学习率 α</div>
                <Slider value={[alpha]} min={0.001} max={0.05} step={0.001} onValueChange={([v]) => setAlpha(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{alpha.toFixed(3)}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">探索率 ε</div>
                <Slider value={[epsilon]} min={0.05} max={1} step={0.05} onValueChange={([v]) => setEpsilon(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{epsilon.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">批次大小</div>
                <Slider value={[batchSize]} min={8} max={64} step={8} onValueChange={([v]) => setBatchSize(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{batchSize}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">回放缓冲容量</div>
                <Slider value={[replayCapacity]} min={500} max={5000} step={500} onValueChange={([v]) => setReplayCapacity(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{replayCapacity}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">目标网络更新间隔</div>
                <Slider value={[targetUpdateInterval]} min={10} max={500} step={10} onValueChange={([v]) => setTargetUpdateInterval(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{targetUpdateInterval}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">训练回合数</div>
                <Slider value={[episodes]} min={50} max={500} step={50} onValueChange={([v]) => setEpisodes(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{episodes}</div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">结果</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-1">
              <div>最终 RMSE（相对 q*）：<span className="font-mono font-semibold">{finalRMSE.toFixed(4)}</span></div>
              <div>回放缓冲当前大小：<span className="font-mono">{result.finalReplaySize}</span></div>
              <div>训练步数：<span className="font-mono">{result.lossHistory.length}</span></div>
            </CardContent>
          </Card>

          <Button onClick={() => setSeed((s) => s + 1)} variant="outline" className="w-full">
            重新随机训练
          </Button>
        </div>
      </div>
    </InteractiveDemo>
  );
}
