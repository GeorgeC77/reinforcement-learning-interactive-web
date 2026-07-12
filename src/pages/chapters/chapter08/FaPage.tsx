import { useState, useMemo } from 'react';
import { Brain, ShieldAlert, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
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
  randomPolicy,
  policyWeightedStateValues,
} from '@/lib/rl/gridworld';
import {
  semiGradientTD,
  actionValueFA,
  dqnGridWorld,
  oneHotFeatures,
  coordinateFeatures,
  polynomialCoordinateFeatures,
  distanceStateFeatures,
  type FeatureMode,
  type ActionValueFeatureMode,
} from '@/lib/rl/fa';
import { mulberry32 } from '@/lib/rl/stochasticApproximation';

type TabKey =
  | 'polynomial'
  | 'semi-gradient'
  | 'feature-design'
  | 'theory'
  | 'action-value'
  | 'dqn';

type PolicyPreset = 'goal' | 'random' | 'right';

const RIGHT_POLICY: (0 | 1 | 2 | 3 | 4)[] = [1, 1, 1, 1, 1, 1, 1, 1, 1];
const GOAL_POLICY: (0 | 1 | 2 | 3 | 4)[] = [1, 2, 2, 1, 2, 2, 1, 1, 4];

function policyFromPreset(preset: PolicyPreset) {
  if (preset === 'right') return deterministicPolicy(RIGHT_POLICY, 5);
  if (preset === 'random') return randomPolicy(9, 5);
  return deterministicPolicy(GOAL_POLICY, 5);
}

function featureVectorText(state: number, mode: FeatureMode, degree?: number): string {
  if (mode === 'onehot') return `[${oneHotFeatures(state, DEFAULT_CONFIG).join(', ')}]`;
  if (mode === 'coordinate') return `[${coordinateFeatures(state, DEFAULT_CONFIG).map((x) => x.toFixed(2)).join(', ')}]`;
  if (mode === 'distance') return `[${distanceStateFeatures(state, DEFAULT_CONFIG).map((x) => x.toFixed(2)).join(', ')}]`;
  return `[${polynomialCoordinateFeatures(state, DEFAULT_CONFIG, degree ?? 2).map((x) => x.toFixed(2)).join(', ')}]`;
}

function rmse(a: number[], b: number[]) {
  return Math.sqrt(a.reduce((sum, v, i) => sum + (v - b[i]) ** 2, 0) / a.length);
}

function weightedRmse(a: number[], b: number[], weights: number[]) {
  const total = weights.reduce((s, w) => s + w, 0);
  if (total === 0) return 0;
  return Math.sqrt(
    a.reduce((sum, v, i) => sum + weights[i] * (v - b[i]) ** 2, 0) / total
  );
}

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
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">第 8 章 值函数近似</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          从表格到函数：用参数化函数逼近大状态空间中的值函数，并分析半梯度 TD、动作值近似与 DQN 的核心机制。
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
          title="半梯度 TD(0) 更新"
          formula={
            <KaTeX
              math={String.raw`w \leftarrow w + \alpha \bigl[ r + \gamma \hat{v}(s', w) - \hat{v}(s, w) \bigr] \nabla_w \hat{v}(s, w)`}
              display
            />
          }
          description="只对目标中的样本梯度进行更新，因此称为半梯度（semi-gradient）方法。"
        />
        <FormulaCard
          title="DQN 目标"
          formula={
            <KaTeX
              math={String.raw`y = r + \gamma \max_{a'} Q(s', a'; w^-), \quad L = \tfrac12 \bigl(y - Q(s,a;w)\bigr)^2`}
              display
            />
          }
          description="目标网络 w^- 稳定 bootstrap 目标，主网络 w 负责当前 Q 估计。"
        />
      </section>

      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as TabKey)}>
        <TabsList className="grid w-full grid-cols-3 md:grid-cols-6 h-auto">
          <TabsTrigger value="polynomial">8.1 数学预热</TabsTrigger>
          <TabsTrigger value="semi-gradient">8.2 半梯度 TD(0)</TabsTrigger>
          <TabsTrigger value="feature-design">8.3 特征设计</TabsTrigger>
          <TabsTrigger value="theory">8.4 理论</TabsTrigger>
          <TabsTrigger value="action-value">8.5 动作值近似</TabsTrigger>
          <TabsTrigger value="dqn">8.6 Deep Q-learning</TabsTrigger>
        </TabsList>

        <TabsContent value="polynomial" className="mt-4">
          <PolynomialFittingDemo />
        </TabsContent>
        <TabsContent value="semi-gradient" className="mt-4">
          <SemiGradientDemo />
        </TabsContent>
        <TabsContent value="feature-design" className="mt-4">
          <FeatureDesignDemo />
        </TabsContent>
        <TabsContent value="theory" className="mt-4">
          <TheorySection />
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
                  <li>线性近似中，one-hot 特征等价于表格法；坐标/多项式/距离特征可实现泛化。</li>
                  <li>半梯度 TD(0) 使用自举目标，更新不是真正的随机梯度下降，但在 mild 条件下收敛。</li>
                  <li>资格迹 λ&gt;0 是 TD(0) 的拓展，λ=0 退化为教材主体。</li>
                  <li>动作值函数近似包括 Sarsa with FA 与 Q-learning with FA，指标应区分策略。</li>
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

// ------------------- 8.1 Polynomial fitting prewarm -------------------
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
  const [seed, setSeed] = useState(1);

  const { weights, predictions } = useMemo(() => {
    const rng = mulberry32(seed);
    const numFeatures = degree + 1;
    let w = new Array(numFeatures).fill(0);
    const n = 50;

    for (let k = 0; k < iterations; k++) {
      const x = rng() * 2 - 1;
      const phi = polynomialFeatures(x, degree);
      const target = trueValue(x) + (rng() * 2 - 1) * noiseStd;
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
    <InteractiveDemo title="8.1 数学预热：曲线拟合如何压缩表格值">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
          <svg width={width} height={height} className="bg-white rounded-lg border border-gray-200">
            <line x1={padding} y1={scaleY(0)} x2={width - padding} y2={scaleY(0)} stroke="#e5e7eb" strokeWidth={1} />
            <path d={truePathD} fill="none" stroke="#22c55e" strokeWidth={2} />
            <path d={predPathD} fill="none" stroke="#2563eb" strokeWidth={2} strokeDasharray="4 2" />
            <text x={width - padding} y={scaleY(1.3)} textAnchor="end" fontSize={10} fill="#22c55e">
              真实值
            </text>
            <text x={width - padding} y={scaleY(1.0)} textAnchor="end" fontSize={10} fill="#2563eb">
              近似值
            </text>
            <text x={padding} y={height - 4} fontSize={10} fill="#6b7280">x</text>
            <text x={4} y={padding - 4} fontSize={10} fill="#6b7280">y</text>
          </svg>
          <p className="mt-4 text-sm text-gray-500 text-center">
            用多项式基函数拟合 sin(πx)。这与 RL 函数近似共享同一思想：用少量参数压缩无限/大表格。
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

          <div className="flex gap-2">
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="flex-1 border rounded px-2 py-1 text-sm"
            />
            <Button onClick={() => setSeed((s) => s + 1)} variant="outline">
              <RefreshCw className="w-4 h-4 mr-1" />
              重新采样
            </Button>
          </div>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- 8.2 Semi-gradient TD(0) -------------------
function SemiGradientDemo() {
  const config = DEFAULT_CONFIG;
  const [policyPreset, setPolicyPreset] = useState<PolicyPreset>('goal');
  const [featureMode, setFeatureMode] = useState<FeatureMode>('coordinate');
  const [degree, setDegree] = useState(2);
  const [alpha, setAlpha] = useState(0.05);
  const [lambda, setLambda] = useState(0);
  const [episodes, setEpisodes] = useState(200);
  const [seed, setSeed] = useState(1);

  const policy = useMemo(() => policyFromPreset(policyPreset), [policyPreset]);

  const result = useMemo(() => {
    return semiGradientTD(policy, config, {
      alpha,
      lambda,
      featureMode,
      polynomialDegree: degree,
      episodes,
      maxSteps: 30,
      seed,
    });
  }, [policy, config, alpha, lambda, featureMode, degree, episodes, seed]);

  const chartData = useMemo(() => {
    return result.valuesHistory.map((v, i) => {
      const unweighted = rmse(v, result.trueValues);
      const weighted = weightedRmse(v, result.trueValues, result.visitCounts);
      return {
        episode: i,
        rmse: unweighted,
        weightedRmse: weighted,
        residual: result.residualHistory[i],
      };
    });
  }, [result]);

  const finalValues = result.valuesHistory[result.valuesHistory.length - 1];
  const finalRMSE = chartData[chartData.length - 1]?.rmse ?? 0;
  const finalWeightedRMSE = chartData[chartData.length - 1]?.weightedRmse ?? 0;

  return (
    <InteractiveDemo title="8.2 目标函数与半梯度 TD(0)">
      <div className="grid lg:grid-cols-[1fr_360px] gap-6">
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
              data={chartData}
              xKey="episode"
              xLabel="回合"
              yLabel="RMSE / residual"
              series={[
                { key: 'rmse', name: '全状态 RMSE', color: '#2563eb' },
                { key: 'weightedRmse', name: '访问加权 RMSE', color: '#22c55e' },
                { key: 'residual', name: 'policy Bellman residual', color: '#ef4444' },
              ]}
              height={220}
            />
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-2">每状态访问次数</h3>
            <div className="grid grid-cols-3 gap-2 text-center">
              {result.visitCounts.map((count, s) => (
                <div
                  key={s}
                  className={`rounded p-2 text-xs ${count === 0 ? 'bg-red-50 text-red-700' : 'bg-gray-50'}`}
                >
                  s{s + 1}: <span className="font-mono font-semibold">{count}</span>
                </div>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-2">
              未访问状态的估计主要来自特征泛化，不能把误差全部归因于 TD 算法。
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm text-gray-700 block mb-1">策略预设</label>
                <Select value={policyPreset} onValueChange={(v) => setPolicyPreset(v as PolicyPreset)}>
                  <SelectTrigger>
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
                    <SelectItem value="distance">距离目标 + 禁区特征</SelectItem>
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
                <div className="text-sm text-gray-600 mb-1">资格迹衰减 λ（教材拓展）</div>
                <Slider value={[lambda]} min={0} max={0.99} step={0.01} onValueChange={([v]) => setLambda(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{lambda.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">训练回合数</div>
                <Slider value={[episodes]} min={10} max={500} step={10} onValueChange={([v]) => setEpisodes(v)} />
                <div className="mt-1 text-center font-mono text-sm text-gray-700">{episodes}</div>
              </div>
              <div className="flex gap-2">
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  className="flex-1 border rounded px-2 py-1 text-sm"
                />
                <Button onClick={() => setSeed((s) => s + 1)} variant="outline" size="sm">
                  <RefreshCw className="w-4 h-4 mr-1" />
                  重新随机
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">结果</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <div>最终全状态 RMSE：<span className="font-mono font-semibold">{finalRMSE.toFixed(4)}</span></div>
              <div>最终访问加权 RMSE：<span className="font-mono font-semibold">{finalWeightedRMSE.toFixed(4)}</span></div>
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

// ------------------- 8.3 Feature design -------------------
function FeatureDesignDemo() {
  const [mode, setMode] = useState<FeatureMode>('coordinate');
  const [degree, setDegree] = useState(2);

  return (
    <InteractiveDemo title="8.3 特征设计与泛化">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>状态</TableHead>
                <TableHead>特征向量 φ(s)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {Array.from({ length: 9 }, (_, s) => (
                <TableRow key={s}>
                  <TableCell className="font-mono">s{s + 1}</TableCell>
                  <TableCell className="font-mono text-xs">{featureVectorText(s, mode, degree)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">特征构造</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Select value={mode} onValueChange={(v) => setMode(v as FeatureMode)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="onehot">One-hot（表格等价，无泛化）</SelectItem>
                  <SelectItem value="coordinate">坐标归一化（平滑泛化）</SelectItem>
                  <SelectItem value="polynomial">坐标多项式（非线性泛化）</SelectItem>
                  <SelectItem value="distance">距离目标 + 禁区（领域知识）</SelectItem>
                </SelectContent>
              </Select>
              {mode === 'polynomial' && (
                <div>
                  <div className="text-sm text-gray-600 mb-1">多项式阶数</div>
                  <Slider value={[degree]} min={1} max={4} step={1} onValueChange={([v]) => setDegree(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{degree}</div>
                </div>
              )}
              <p className="text-sm text-gray-600">
                不同特征决定相近状态是否获得相近的值估计，从而影响泛化能力与逼近误差。
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- 8.4 Theory -------------------
function TheorySection() {
  return (
    <InteractiveDemo title="8.4 投影、Bellman 误差与收敛">
      <div className="grid md:grid-cols-2 gap-4">
        <FormulaCard
          title="投影 Bellman 方程（PBE）"
          formula={
            <KaTeX
              math={String.raw`\Pi_\mathcal{F} T_\pi V_w = V_w`}
              display
            />
          }
          description="Π_F 把任意值函数投影到函数近似子空间，半梯度 TD 寻找 PBE 的不动点。"
        />
        <FormulaCard
          title="均方 Bellman 误差"
          formula={
            <KaTeX
              math={String.raw`\text{MSBE}(w) = \mathbb{E}\bigl[(r + \gamma \hat{v}(s',w) - \hat{v}(s,w))^2\bigr]`}
              display
            />
          }
          description="半梯度 TD(0) 不是 MSBE 的纯梯度下降，但可收敛到 PBE 的近似解。"
        />
        <FormulaCard
          title="收敛条件"
          formula={
            <KaTeX
              math={String.raw`\alpha_t \to 0, \quad \sum_t \alpha_t = \infty, \quad \sum_t \alpha_t^2 < \infty`}
              display
            />
          }
          description="线性函数近似、on-policy 数据与递减步长下，半梯度 TD(0) 可收敛。"
        />
        <FormulaCard
          title="值函数近似 vs 表格法"
          formula={
            <KaTeX
              math={String.raw`\hat{v}(s,w) \approx v_\pi(s), \quad \forall s \in \mathcal{S}`}
              display
            />
          }
          description="表格法是 one-hot 特征下的特例；函数近似把表格的一列压缩为参数向量。"
        />
      </div>
    </InteractiveDemo>
  );
}

// ------------------- 8.5 Action-value FA -------------------
function ActionValueFADemo() {
  const config = DEFAULT_CONFIG;
  const qStar = useMemo(() => estimateTrueActionValues(config), [config]);
  const [algorithm, setAlgorithm] = useState<'sarsa' | 'qlearning'>('sarsa');
  const [featureMode, setFeatureMode] = useState<ActionValueFeatureMode>('shared');
  const [alpha, setAlpha] = useState(0.05);
  const [epsilon, setEpsilon] = useState(0.3);
  const [episodes, setEpisodes] = useState(200);
  const [seed, setSeed] = useState(1);

  const result = useMemo(() => {
    return actionValueFA(config, {
      alpha,
      epsilon,
      gamma: config.gamma,
      episodes,
      maxSteps: 30,
      featureMode,
      algorithm,
      seed,
    });
  }, [config, algorithm, featureMode, alpha, epsilon, episodes, seed]);

  const finalQ = result.qHistory[result.qHistory.length - 1];
  const finalPolicy = useMemo(() => greedyPolicy(finalQ), [finalQ]);
  const behaviorPolicy = result.behaviorPolicyHistory[result.behaviorPolicyHistory.length - 1];
  const behaviorValues = useMemo(() => policyWeightedStateValues(finalQ, behaviorPolicy), [finalQ, behaviorPolicy]);
  const greedyValues = useMemo(() => actionValueToStateValue(finalQ), [finalQ]);
  const finalRMSE = useMemo(() => qTableRMSE(finalQ, qStar), [finalQ, qStar]);

  const chartData = useMemo(() => {
    return result.residualHistory.map((res, i) => ({
      episode: i,
      residual: res,
      rmse: qTableRMSE(result.qHistory[i], qStar),
    }));
  }, [result, qStar]);

  const last = result.lastUpdate;

  return (
    <InteractiveDemo title="8.5 动作值函数近似 q(s,a,w)">
      <div className="grid lg:grid-cols-[1fr_360px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld
              config={config}
              policy={finalPolicy}
              values={algorithm === 'qlearning' ? greedyValues : behaviorValues}
              showValues
              className="max-w-full"
            />
            <p className="mt-3 text-sm text-gray-500 text-center">
              {algorithm === 'qlearning'
                ? 'Q-learning with FA：显示贪心派生价值'
                : 'Sarsa with FA：显示当前 ε-soft 行为策略价值'}
            </p>
          </div>
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <LineChart
              data={chartData}
              xKey="episode"
              xLabel="回合"
              yLabel={algorithm === 'qlearning' ? '最优 Bellman 残差 / q* RMSE' : '同策略 Bellman 残差 / q* RMSE'}
              series={[
                { key: 'residual', name: algorithm === 'qlearning' ? 'Bellman optimality residual' : 'behavior-policy Bellman residual', color: '#ef4444' },
                { key: 'rmse', name: 'q* RMSE（参考）', color: '#2563eb' },
              ]}
              height={220}
            />
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
                    <SelectValue />
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
                    <SelectValue />
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
              <div className="flex gap-2">
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  className="flex-1 border rounded px-2 py-1 text-sm"
                />
                <Button onClick={() => setSeed((s) => s + 1)} variant="outline" size="sm">
                  <RefreshCw className="w-4 h-4 mr-1" />
                  重新随机
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">结果与说明</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <div>
                {algorithm === 'qlearning'
                  ? 'Q-learning 显示 Bellman optimality residual 与贪心派生价值。'
                  : 'Sarsa 固定 ε 时显示 behavior-policy Bellman residual；q* RMSE 仅作为参考，不能断言收敛到 q*。'}
              </div>
              <div>最终 q* RMSE：<span className="font-mono font-semibold">{finalRMSE.toFixed(4)}</span></div>
              <div>权重维度：<span className="font-mono">{result.weightsHistory[0].length}</span></div>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- 8.6 DQN -------------------
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
  const [seed, setSeed] = useState(1);

  const result = useMemo(() => {
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
      seed,
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
    <InteractiveDemo title="8.6 Deep Q-learning / DQN：神经网络 + 经验回放 + 目标网络">
      <div className="grid lg:grid-cols-[1fr_360px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld config={config} policy={finalPolicy} values={finalValues} showValues className="max-w-full" />
            <p className="mt-3 text-sm text-gray-500 text-center">训练结束后 DQN 的贪心策略与状态值</p>
          </div>

          <div className="bg-blue-50 rounded-xl p-4 border border-blue-100 text-sm text-gray-700">
            <h3 className="font-semibold text-blue-800 mb-2">DQN 训练流程</h3>
            <ol className="list-decimal pl-5 space-y-1">
              <li>与环境交互得到 transition (s, a, r, s&apos;, done)；</li>
              <li>存入经验回放 replay buffer；</li>
              <li>从 buffer 中均匀抽取 mini-batch；</li>
              <li>用目标网络计算 y = r + γ max_a&apos; Q(s&apos;,a&apos;; w⁻)；</li>
              <li>用主网络计算 Q(s,a; w)；</li>
              <li>损失 L = ½(y − Q)²，梯度更新主网络；</li>
              <li>每隔固定步数把主网络参数复制给目标网络。</li>
            </ol>
            <p className="mt-2 text-xs text-gray-600">
              经验回放通过随机抽样降低相邻经验的时序相关性，并提高历史样本复用率。
            </p>
          </div>

          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <LineChart
              data={lossData}
              xKey="step"
              xLabel="训练步"
              yLabel="½(y − Q)²"
              series={[{ key: 'loss', name: 'DQN Loss', color: '#ef4444' }]}
              height={220}
            />
          </div>
          {result.lastBatch.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <h3 className="font-semibold text-gray-800 mb-2 text-sm">
                最近一次 mini-batch（共 {result.lastBatch.length} 条）
              </h3>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-10">s</TableHead>
                      <TableHead className="w-10">a</TableHead>
                      <TableHead className="w-10">r</TableHead>
                      <TableHead className="w-10">s&apos;</TableHead>
                      <TableHead className="w-14">done</TableHead>
                      <TableHead>target y</TableHead>
                      <TableHead>Q(s,a)</TableHead>
                      <TableHead>loss</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.lastBatch.map((row, i) => (
                      <TableRow key={i} className="bg-blue-50/60">
                        <TableCell className="font-mono">{row.state + 1}</TableCell>
                        <TableCell className="font-mono">{ACTION_NAMES[row.action]}</TableCell>
                        <TableCell className="font-mono">{row.reward.toFixed(2)}</TableCell>
                        <TableCell className="font-mono">{row.nextState + 1}</TableCell>
                        <TableCell className="font-mono">{row.done ? '是' : '否'}</TableCell>
                        <TableCell className="font-mono">{row.target.toFixed(3)}</TableCell>
                        <TableCell className="font-mono">{row.prediction.toFixed(3)}</TableCell>
                        <TableCell className="font-mono">{row.loss.toFixed(4)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}
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
              <div className="flex gap-2">
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  className="flex-1 border rounded px-2 py-1 text-sm"
                />
                <Button onClick={() => setSeed((s) => s + 1)} variant="outline" size="sm">
                  <RefreshCw className="w-4 h-4 mr-1" />
                  重新随机
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">结果</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-1">
              <div>最终 q* RMSE：<span className="font-mono font-semibold">{finalRMSE.toFixed(4)}</span></div>
              <div>回放缓冲当前大小：<span className="font-mono">{result.finalReplaySize}</span></div>
              <div>训练步数：<span className="font-mono">{result.lossHistory.length}</span></div>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}
