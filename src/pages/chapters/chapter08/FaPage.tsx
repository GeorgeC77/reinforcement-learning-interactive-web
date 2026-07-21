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
  EPISODIC_PATH_CONFIG,
  ACTION_NAMES,
  deterministicPolicy,
  greedyPolicy,
  actionValueToStateValue,
  estimateTrueActionValues,
  qTableRMSE,
  randomPolicy,
  policyWeightedStateValues,
  isTerminal,
  optimalBellmanResidualQ,
  greedyActionAgreement,
  step,
  type GridWorldConfig,
  type Policy,
  type Action,
} from '@/lib/rl/gridworld';
import {
  semiGradientTD,
  actionValueFA,
  dqnGridWorld,
  oneHotFeatures,
  coordinateFeatures,
  polynomialCoordinateFeatures,
  distanceStateFeatures,
  stationaryDistribution,
  lstdFromTrajectory,
  rlsTD,
  type FeatureMode,
  type ActionValueFeatureMode,
  type EpsilonScheduleMode,
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
type TaskType = 'continuing' | 'episodic';
type WeightMode = 'uniform' | 'empirical' | 'stationary';

const RIGHT_POLICY: Action[] = [1, 1, 1, 1, 1, 1, 1, 1, 1];
const GOAL_POLICY: Action[] = [1, 2, 2, 1, 2, 2, 1, 1, 4];
const H_OPTIONS = [10, 20, 30, 50, 100, 200];

function configForTask(taskType: TaskType): GridWorldConfig {
  return taskType === 'continuing' ? DEFAULT_CONFIG : EPISODIC_PATH_CONFIG;
}

function policyFromPreset(preset: PolicyPreset, numStates: number = 9, numActions: number = 5): Policy {
  if (preset === 'right') return deterministicPolicy(RIGHT_POLICY, numActions);
  if (preset === 'random') return randomPolicy(numStates, numActions);
  return deterministicPolicy(GOAL_POLICY, numActions);
}

function featureVectorText(state: number, mode: FeatureMode, degree: number, config: GridWorldConfig): string {
  if (mode === 'onehot') return `[${oneHotFeatures(state, config).join(', ')}]`;
  if (mode === 'coordinate') return `[${coordinateFeatures(state, config).map((x) => x.toFixed(2)).join(', ')}]`;
  if (mode === 'distance') return `[${distanceStateFeatures(state, config).map((x) => x.toFixed(2)).join(', ')}]`;
  return `[${polynomialCoordinateFeatures(state, config, degree).map((x) => x.toFixed(2)).join(', ')}]`;
}

function featuresForState(state: number, mode: FeatureMode, degree: number, config: GridWorldConfig): number[] {
  if (mode === 'onehot') return oneHotFeatures(state, config);
  if (mode === 'coordinate') return coordinateFeatures(state, config);
  if (mode === 'distance') return distanceStateFeatures(state, config);
  return polynomialCoordinateFeatures(state, config, degree);
}

function rmse(a: number[], b: number[]) {
  if (a.length === 0) return 0;
  return Math.sqrt(a.reduce((sum, v, i) => sum + (v - b[i]) ** 2, 0) / a.length);
}

function weightedObjective(
  a: number[],
  b: number[],
  weights: number[]
) {
  const total = weights.reduce((s, w) => s + w, 0);
  if (total === 0) return { j: 0, contributions: a.map(() => 0) };
  const squared = a.map((v, i) => (v - b[i]) ** 2);
  const contributions = squared.map((v, i) => weights[i] * v);
  const j = contributions.reduce((s, v) => s + v, 0) / total;
  return { j, contributions: contributions.map((v) => v / total) };
}

function weightedRmse(a: number[], b: number[], weights: number[]) {
  const total = weights.reduce((s, w) => s + w, 0);
  if (total === 0) return 0;
  return Math.sqrt(
    a.reduce((sum, v, i) => sum + weights[i] * (v - b[i]) ** 2, 0) / total
  );
}

function unvisitedRmse(a: number[], b: number[], visits: number[]) {
  const idx = visits.map((v, i) => (v === 0 ? i : -1)).filter((i) => i >= 0);
  if (idx.length === 0) return 0;
  return Math.sqrt(idx.reduce((sum, i) => sum + (a[i] - b[i]) ** 2, 0) / idx.length);
}



function computeUpdateEffect(
  selectedState: number,
  policy: Policy,
  config: GridWorldConfig,
  featureMode: FeatureMode,
  degree: number,
  alpha: number,
  seed: number,
  values: number[]
): { effect: number[]; action: Action; reward: number; nextState: number; delta: number } {
  const rng = mulberry32(seed + selectedState * 12345);
  const phi =
    featureMode === 'onehot'
      ? (s: number) => oneHotFeatures(s, config)
      : featureMode === 'coordinate'
      ? (s: number) => coordinateFeatures(s, config)
      : featureMode === 'distance'
      ? (s: number) => distanceStateFeatures(s, config)
      : (s: number) => polynomialCoordinateFeatures(s, config, degree);

  // Use the policy at the selected state, deterministic tie-breaking via rng
  const dist = policy[selectedState];
  let r = rng();
  let cum = 0;
  let action: Action = 0;
  for (let a = 0; a < dist.length; a++) {
    cum += dist[a];
    if (r <= cum) {
      action = a as Action;
      break;
    }
  }

  const { nextState, reward } = step(selectedState, action, config);
  const vS = values[selectedState];
  const vNext = isTerminal(nextState, config) ? 0 : values[nextState];
  const delta = reward + config.gamma * vNext - vS;
  const phiS = phi(selectedState);

  const effect = values.map((_, t) => alpha * delta * dotProduct(phi(t), phiS));
  return { effect, action, reward, nextState, delta };
}

function dotProduct(a: number[], b: number[]) {
  return a.reduce((sum, v, i) => sum + v * (b[i] ?? 0), 0);
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
        <TabsList className="grid w-full grid-cols-2 md:grid-cols-3 lg:grid-cols-6 h-auto">
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
                  <li>资格迹 λ&gt;0 是 TD(0) 的拓展，λ=0 退化为标准 TD(0)。</li>
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
              <SliderControl label="多项式阶数 d" value={degree} min={0} max={8} step={1} onChange={setDegree} />
              <SliderControl label="学习率 α" value={alpha} min={0.001} max={0.2} step={0.001} onChange={setAlpha} />
              <SliderControl label="训练迭代次数" value={iterations} min={0} max={1000} step={50} onChange={setIterations} />
              <SliderControl label="目标噪声" value={noiseStd} min={0} max={0.5} step={0.05} onChange={setNoiseStd} />
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

          <SeedControl seed={seed} onChange={setSeed} />
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- 8.2 Semi-gradient TD(0) -------------------
function SemiGradientDemo() {
  const [policyPreset, setPolicyPreset] = useState<PolicyPreset>('goal');
  const [featureMode, setFeatureMode] = useState<FeatureMode>('coordinate');
  const [degree, setDegree] = useState(2);
  const [alpha, setAlpha] = useState(0.05);
  const [lambda, setLambda] = useState(0);
  const [episodes, setEpisodes] = useState(200);
  const [maxSteps, setMaxSteps] = useState(30); // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
  const [seed, setSeed] = useState(1);
  const [weightMode, setWeightMode] = useState<WeightMode>('uniform');

  const config = DEFAULT_CONFIG;
  const policy = useMemo(() => policyFromPreset(policyPreset), [policyPreset]);
  const stationaryDist = useMemo(() => stationaryDistribution(policy, config), [policy, config]);

  const result = useMemo(() => {
    return semiGradientTD(policy, config, {
      alpha,
      lambda,
      featureMode,
      polynomialDegree: degree,
      episodes,
      maxSteps,
      seed,
    });
  }, [policy, config, alpha, lambda, featureMode, degree, episodes, maxSteps, seed]);

  const weightsForMode = useMemo(() => {
    const numStates = config.rows * config.cols;
    if (weightMode === 'uniform') return new Array(numStates).fill(1 / numStates);
    if (weightMode === 'empirical') {
      const total = result.visitCounts.reduce((s, v) => s + v, 0);
      return total === 0 ? new Array(numStates).fill(1 / numStates) : result.visitCounts.map((v) => v / total);
    }
    return stationaryDist;
  }, [weightMode, result.visitCounts, stationaryDist, config]);

  function weightsAtEpisode(i: number): number[] {
    const numStates = config.rows * config.cols;
    if (weightMode === 'uniform') return new Array(numStates).fill(1 / numStates);
    if (weightMode === 'empirical') {
      const counts = result.visitCountsHistory[i] ?? result.visitCounts;
      const total = counts.reduce((s, v) => s + v, 0);
      return total === 0 ? new Array(numStates).fill(1 / numStates) : counts.map((v) => v / total);
    }
    return stationaryDist;
  }

  const valueErrorData = useMemo(() => {
    return result.valuesHistory.map((v, i) => ({
      episode: i,
      rmse: rmse(v, result.trueValues),
      weightedRmse: weightedRmse(v, result.trueValues, weightsAtEpisode(i)),
    }));
  }, [result, weightMode, stationaryDist, config]);

  const fixedPointData = useMemo(() => {
    return result.residualHistory.map((res, i) => ({ episode: i, residual: res }));
  }, [result]);

  const objectiveData = useMemo(() => {
    return result.valuesHistory.map((v, i) => {
      const { j } = weightedObjective(v, result.trueValues, weightsAtEpisode(i));
      return { episode: i, objective: j };
    });
  }, [result, weightMode, stationaryDist, config]);

  const finalValues = result.valuesHistory[result.valuesHistory.length - 1];
  const finalObj = weightedObjective(finalValues, result.trueValues, weightsForMode);
  const distributionData = weightsForMode.map((d, s) => ({
    state: `s${s + 1}`,
    d,
    squaredError: (finalValues[s] - result.trueValues[s]) ** 2,
    contribution: finalObj.contributions[s],
  }));

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

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">8.2.1 目标函数 J(w) = Σ_s d(s)[v_π(s) − v̂(s,w)]²</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {(['uniform', 'empirical', 'stationary'] as WeightMode[]).map((m) => (
                  <Button
                    key={m}
                    variant={weightMode === m ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setWeightMode(m)}
                  >
                    {m === 'uniform' ? 'uniform' : m === 'empirical' ? 'empirical visitation' : 'stationary'}
                  </Button>
                ))}
              </div>
              <div className="grid md:grid-cols-2 gap-4">
                <SimpleBarChart
                  data={distributionData.map((d) => ({ label: d.state, value: d.d }))}
                  title="状态分布 d(s)"
                  color="#3b82f6"
                />
                <SimpleBarChart
                  data={distributionData.map((d) => ({ label: d.state, value: d.squaredError }))}
                  title="每状态 squared error"
                  color="#ef4444"
                />
              </div>
              <div className="grid md:grid-cols-3 gap-3 text-sm">
                <div className="bg-gray-50 rounded p-2 text-center">
                  <div className="text-gray-500">总目标 J(w)</div>
                  <div className="font-mono font-semibold">{finalObj.j.toFixed(4)}</div>
                </div>
                <div className="bg-gray-50 rounded p-2 text-center">
                  <div className="text-gray-500">全状态 RMSE</div>
                  <div className="font-mono font-semibold">{rmse(finalValues, result.trueValues).toFixed(4)}</div>
                </div>
                <div className="bg-gray-50 rounded p-2 text-center">
                  <div className="text-gray-500">访问加权 RMSE</div>
                  <div className="font-mono font-semibold">{weightedRmse(finalValues, result.trueValues, weightsForMode).toFixed(4)}</div>
                </div>
              </div>
              <p className="text-xs text-gray-600">
                访问加权 RMSE 与 empirical visitation 加权的目标函数同量纲。未访问状态的误差若被放大，说明拟合过度依赖特征泛化。
              </p>
            </CardContent>
          </Card>

          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200 space-y-6">
            <div data-testid="fa-value-error-chart">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">Value approximation error</h4>
              <LineChart
                data={valueErrorData}
                xKey="episode"
                xLabel="回合"
                yLabel="RMSE"
                series={[
                  { key: 'rmse', name: 'full-state RMSE', color: '#2563eb' },
                  { key: 'weightedRmse', name: `${weightMode} weighted RMSE`, color: '#22c55e' },
                ]}
                height={180}
              />
            </div>
            <div data-testid="fa-fixed-point-chart">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">Fixed-point error</h4>
              <LineChart
                data={fixedPointData}
                xKey="episode"
                xLabel="回合"
                yLabel="‖TπV - V‖∞"
                series={[{ key: 'residual', name: 'policy Bellman residual', color: '#ef4444' }]}
                height={160}
              />
            </div>
            <div data-testid="fa-objective-chart">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">Objective</h4>
              <LineChart
                data={objectiveData}
                xKey="episode"
                xLabel="回合"
                yLabel={`J(w) (${weightMode})`}
                series={[{ key: 'objective', name: `J(w) (${weightMode})`, color: '#f59e0b' }]}
                height={160}
              />
            </div>
            <p className="text-xs text-gray-600">
              历史曲线按当前选定的分布（{weightMode}）逐回合评价；其中 empirical 使用对应时刻的累积访问次数。
            </p>
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
                <p className="text-xs text-amber-600">反例：在右边界反复碰撞。</p>
              )}

              <Select value={featureMode} onValueChange={(v) => setFeatureMode(v as FeatureMode)}>
                <SelectTrigger data-testid="fa-feature-select">
                  <SelectValue placeholder="选择特征" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="onehot">One-hot（表格等价）</SelectItem>
                  <SelectItem value="coordinate">坐标归一化</SelectItem>
                  <SelectItem value="polynomial">坐标多项式</SelectItem>
                  <SelectItem value="distance">距离目标 + 禁区特征</SelectItem>
                </SelectContent>
              </Select>

              {featureMode === 'polynomial' && (
                <SliderControl label="多项式阶数" value={degree} min={1} max={4} step={1} onChange={setDegree} />
              )}
              <SliderControl label="学习率 α" value={alpha} min={0.001} max={0.2} step={0.001} onChange={setAlpha} />
              <SliderControl label="资格迹衰减 λ" value={lambda} min={0} max={0.99} step={0.01} onChange={setLambda} />
              <SliderControl label="训练回合数" value={episodes} min={10} max={500} step={10} onChange={setEpisodes} />
              <HorizonControl value={maxSteps} onChange={setMaxSteps} />
              <SeedControl seed={seed} onChange={setSeed} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">结果</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <div>最终全状态 RMSE：<span className="font-mono font-semibold">{valueErrorData[valueErrorData.length - 1]?.rmse.toFixed(4)}</span></div>
              <div>最终访问加权 RMSE：<span className="font-mono font-semibold">{valueErrorData[valueErrorData.length - 1]?.weightedRmse.toFixed(4)}</span></div>
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

// ------------------- 8.3 Feature design as comparison -------------------
function FeatureDesignDemo() {
  const [policyPreset, setPolicyPreset] = useState<PolicyPreset>('random');
  const [alpha, setAlpha] = useState(0.05);
  const [lambda, setLambda] = useState(0);
  const [episodes, setEpisodes] = useState(100);
  const [maxSteps, setMaxSteps] = useState(30); // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
  const [seed, setSeed] = useState(1);
  const [selectedState, setSelectedState] = useState<number | null>(null);

  const config = DEFAULT_CONFIG;
  const policy = useMemo(() => policyFromPreset(policyPreset), [policyPreset]);

  const modes: { key: FeatureMode; label: string; degree?: number }[] = [
    { key: 'onehot', label: 'one-hot' },
    { key: 'coordinate', label: 'coordinate' },
    { key: 'polynomial', label: 'polynomial (d=2)', degree: 2 },
    { key: 'distance', label: 'distance/domain' },
  ];

  const results = useMemo(() => {
    return modes.map((m) => {
      const res = semiGradientTD(policy, config, {
        alpha,
        lambda,
        featureMode: m.key,
        polynomialDegree: m.degree,
        episodes,
        maxSteps,
        seed,
      });
      const final = res.valuesHistory[res.valuesHistory.length - 1];
      return {
        ...m,
        result: res,
        final,
        dim: res.weightsHistory[0].length,
        rmse: rmse(final, res.trueValues),
        weightedRmse: weightedRmse(final, res.trueValues, res.visitCounts),
        residual: res.residualHistory[res.residualHistory.length - 1],
        unvisitedRmse: unvisitedRmse(final, res.trueValues, res.visitCounts),
      };
    });
  }, [policy, config, alpha, lambda, episodes, maxSteps, seed]);

  const selectedEffect = useMemo(() => {
    if (selectedState === null) return null;
    return results.map((r) => {
      const effectData = computeUpdateEffect(selectedState, policy, config, r.key, r.degree ?? 2, alpha, seed, r.final);
      return { key: r.key, label: r.label, ...effectData };
    });
  }, [selectedState, results, policy, config, alpha, seed]);

  const curveData = useMemo(() => {
    const len = results[0].result.valuesHistory.length;
    return Array.from({ length: len }, (_, i) => {
      const point: Record<string, number> = { episode: i };
      results.forEach((r) => {
        point[r.key] = rmse(r.result.valuesHistory[i], r.result.trueValues);
      });
      return point;
    });
  }, [results]);

  return (
    <InteractiveDemo title="8.3 特征设计：同一训练预算下的并排对比">
      <div className="space-y-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">实验设置</CardTitle>
          </CardHeader>
          <CardContent className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Select value={policyPreset} onValueChange={(v) => setPolicyPreset(v as PolicyPreset)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="goal">goal policy</SelectItem>
                <SelectItem value="random">random policy</SelectItem>
                <SelectItem value="right">right policy</SelectItem>
              </SelectContent>
            </Select>
            <SliderControl label="学习率 α" value={alpha} min={0.001} max={0.2} step={0.001} onChange={setAlpha} />
            <SliderControl label="λ" value={lambda} min={0} max={0.99} step={0.01} onChange={setLambda} />
            <SliderControl label="回合数" value={episodes} min={10} max={300} step={10} onChange={setEpisodes} />
            <HorizonControl value={maxSteps} onChange={setMaxSteps} />
            <SeedControl seed={seed} onChange={setSeed} />
          </CardContent>
        </Card>

        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>特征</TableHead>
                <TableHead>维度</TableHead>
                <TableHead>RMSE</TableHead>
                <TableHead>访问加权 RMSE</TableHead>
                <TableHead>未访问 RMSE</TableHead>
                <TableHead>Bellman residual</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {results.map((r) => (
                <TableRow key={r.key}>
                  <TableCell className="font-medium">{r.label}</TableCell>
                  <TableCell className="font-mono">{r.dim}</TableCell>
                  <TableCell className="font-mono">{r.rmse.toFixed(4)}</TableCell>
                  <TableCell className="font-mono">{r.weightedRmse.toFixed(4)}</TableCell>
                  <TableCell className="font-mono">{r.unvisitedRmse.toFixed(4)}</TableCell>
                  <TableCell className="font-mono">{r.residual.toFixed(4)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
          <LineChart
            data={curveData}
            xKey="episode"
            xLabel="回合"
            yLabel="RMSE"
            series={results.map((r, i) => ({ key: r.key, name: r.label, color: ['#2563eb', '#22c55e', '#f59e0b', '#a855f7'][i] }))}
            height={220}
          />
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {results.map((r) => (
            <div key={r.key} className="flex flex-col items-center bg-white rounded-xl border border-gray-200 p-3">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">{r.label}</h4>
              <GridWorld
                config={config}
                values={r.final}
                showValues
                highlightState={selectedState ?? undefined}
                onCellClick={(s) => setSelectedState(s)}
                className="max-w-full"
              />
            </div>
          ))}
        </div>

        {selectedState !== null && selectedEffect && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">
                选中 s{selectedState + 1}：模拟一次 TD 更新对其他状态的泛化影响
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600 mb-3">
                颜色深浅表示：若在当前估计上，对 s{selectedState + 1} 执行一次半梯度 TD 更新，其他状态的预测值改变量。
                one-hot 仅自身变化；共享特征会使邻近状态一起变化。
              </p>
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                {selectedEffect.map((eff) => (
                  <div key={eff.key} className="flex flex-col items-center">
                    <h5 className="text-xs font-semibold text-gray-600 mb-1">{eff.label}</h5>
                    <GridWorld
                      config={config}
                      values={eff.effect}
                      showValues
                      highlightState={selectedState}
                      className="max-w-full"
                    />
                    <div className="text-xs text-gray-500 mt-1">
                      a={ACTION_NAMES[eff.action]}, r={eff.reward.toFixed(1)}, δ={eff.delta.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">特征向量示例</h3>
          <div className="grid md:grid-cols-2 gap-4 text-xs">
            {results.map((r) => (
              <div key={r.key} className="bg-gray-50 rounded p-2">
                <div className="font-medium mb-1">{r.label}</div>
                <div className="font-mono break-all">{featureVectorText(4, r.key, r.degree ?? 2, config)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- 8.4 Theory -------------------
function TheorySection() {
  const [active, setActive] = useState('value-error');
  const [policyPreset, setPolicyPreset] = useState<PolicyPreset>('goal');
  const [featureMode, setFeatureMode] = useState<FeatureMode>('coordinate');
  const [degree, setDegree] = useState(2);
  const [lstdSeed, setLstdSeed] = useState(1);

  const config = DEFAULT_CONFIG;
  const policy = useMemo(() => policyFromPreset(policyPreset), [policyPreset]);
  const lstdResult = useMemo(
    () => lstdFromTrajectory(policy, config, { featureMode, polynomialDegree: degree, seed: lstdSeed }),
    [policy, config, featureMode, degree, lstdSeed]
  );
  const tdResult = useMemo(
    () => semiGradientTD(policy, config, { alpha: 0.05, lambda: 0, featureMode, polynomialDegree: degree, episodes: 200, maxSteps: 30, seed: lstdSeed }), // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
    [policy, config, featureMode, degree, lstdSeed]
  );
  const rlsResult = useMemo(
    () => rlsTD(policy, config, { featureMode, polynomialDegree: degree, episodes: 200, maxSteps: 30, seed: lstdSeed }), // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
    [policy, config, featureMode, degree, lstdSeed]
  );
  const rlsChartData = useMemo(() => {
    const numStates = config.rows * config.cols;
    const lstdRmse = lstdResult.ok
      ? rmse(
          Array.from({ length: numStates }, (_, s) => {
            const phi = featuresForState(s, featureMode, degree, config);
            return phi.reduce((sum, x, i) => sum + x * (lstdResult.w[i] ?? 0), 0);
          }),
          rlsResult.trueValues
        )
      : null;
    return rlsResult.valuesHistory.map((v, i) => ({
      episode: i,
      rls: rmse(v, rlsResult.trueValues),
      td: rmse(tdResult.valuesHistory[i] ?? v, tdResult.trueValues),
      ...(lstdRmse !== null ? { lstd: lstdRmse } : {}),
    }));
  }, [rlsResult, tdResult, lstdResult, config, featureMode, degree]);

  return (
    <InteractiveDemo title="8.4 理论、投影与收敛">
      <Tabs value={active} onValueChange={setActive}>
        <TabsList className="grid w-full grid-cols-2 md:grid-cols-3 lg:grid-cols-6 h-auto">
          <TabsTrigger value="value-error">Value-error objective</TabsTrigger>
          <TabsTrigger value="bellman-error">Bellman error</TabsTrigger>
          <TabsTrigger value="pbe">Projected BE</TabsTrigger>
          <TabsTrigger value="pbe-equation">Projected BE equation</TabsTrigger>
          <TabsTrigger value="lstd">LSTD</TabsTrigger>
          <TabsTrigger value="rls">Recursive LS</TabsTrigger>
        </TabsList>

        <TabsContent value="value-error" className="mt-4 space-y-4">
          <FormulaCard
            title="Value-error objective"
            formula={<KaTeX math={String.raw`J(w) = \sum_s d(s)\bigl[v_\pi(s) - \hat{v}(s,w)\bigr]^2`} display />}
            description="用状态分布 d(s) 加权均方误差。demo 中可切换 uniform、empirical visitation 或 stationary distribution。"
          />
          <p className="text-sm text-gray-700">
            最小化 J(w) 相当于在函数子空间里把真实值函数 v_π 投影到以 d(s) 为内积权重的近似函数上。
          </p>
        </TabsContent>

        <TabsContent value="bellman-error" className="mt-4 space-y-4">
          <FormulaCard
            title="Mean Squared Bellman Error"
            formula={<KaTeX math={String.raw`\text{MSBE}(w) = \mathbb{E}\bigl[\bigl(r + \gamma \hat{v}(s',w) - \hat{v}(s,w)\bigr)^2\bigr]`} display />}
            description="Bellman error 度量当前估计与 Bellman 算子 T_π 输出之间的差距。"
          />
          <p className="text-sm text-gray-700">
            直接最小化 MSBE 需要对目标也求梯度（包含 w），半梯度 TD 只走了一半，因此称为 semi-gradient。
          </p>
        </TabsContent>

        <TabsContent value="pbe" className="mt-4 space-y-4">
          <FormulaCard
            title="Projected Bellman Error"
            formula={<KaTeX math={String.raw`\text{PBE}(w) = \|\Pi_\mathcal{F} T_\pi \hat{v}(\cdot,w) - \hat{v}(\cdot,w)\|_d^2`} display />}
            description="先把 Bellman 算子输出投影回函数子空间，再与当前估计比较。"
          />
          <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
            <svg width={400} height={160} className="mx-auto bg-white rounded-lg border border-gray-200">
              <text x={40} y={80} textAnchor="middle" fontSize={12} fill="#374151">v̂(w)</text>
              <circle cx={40} cy={80} r={4} fill="#2563eb" />
              <text x={200} y={40} textAnchor="middle" fontSize={12} fill="#374151">T_π v̂(w)</text>
              <circle cx={200} cy={40} r={4} fill="#ef4444" />
              <text x={320} y={120} textAnchor="middle" fontSize={12} fill="#374151">Π_F T_π v̂(w)</text>
              <circle cx={320} cy={120} r={4} fill="#22c55e" />
              <path d="M 55 75 L 185 50" stroke="#ef4444" strokeWidth={2} markerEnd="url(#arrow)" />
              <path d="M 210 55 L 305 105" stroke="#22c55e" strokeWidth={2} strokeDasharray="4 2" markerEnd="url(#arrow)" />
              <path d="M 320 105 L 55 90" stroke="#2563eb" strokeWidth={2} strokeDasharray="4 2" markerEnd="url(#arrow)" />
              <defs>
                <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                  <path d="M0,0 L0,6 L9,3 z" fill="#6b7280" />
                </marker>
              </defs>
            </svg>
            <p className="text-center text-sm text-gray-600 mt-2">
              先应用 Bellman 算子，再投影回子空间；半梯度 TD 寻找该闭合点。
            </p>
          </div>
        </TabsContent>

        <TabsContent value="pbe-equation" className="mt-4 space-y-4">
          <FormulaCard
            title="Projected Bellman Equation"
            formula={<KaTeX math={String.raw`\hat{v}(\cdot,w) = \Pi_\mathcal{F} T_\pi \hat{v}(\cdot,w)`} display />}
            description="半梯度 TD(0) 的不动点方程。解不一定最小化未投影的 Bellman error。"
          />
          <p className="text-sm text-gray-700">
            这个方程说明：好的近似不是让 Bellman residual 为 0（那通常不可表示），而是让投影后的残差为 0。
          </p>
        </TabsContent>

        <TabsContent value="lstd" className="mt-4 space-y-4">
          <FormulaCard
            title="LSTD 闭合解"
            formula={
              <KaTeX
                math={String.raw`\hat{A}_t = \sum_{k=0}^t \phi_k(\phi_k - \gamma \phi_{k+1})^\top, \quad \hat{b}_t = \sum_{k=0}^t r_{k+1}\phi_k, \quad \hat{w}_t = \hat{A}_t^{-1}\hat{b}_t`}
                display
              />
            }
            description="最小二乘时间差分直接求解析解，不需要学习率，但计算和存储 A 矩阵的成本更高。"
          />
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">LSTD 演示设置</CardTitle>
            </CardHeader>
            <CardContent className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Select value={policyPreset} onValueChange={(v) => setPolicyPreset(v as PolicyPreset)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="goal">goal policy</SelectItem>
                  <SelectItem value="random">random policy</SelectItem>
                  <SelectItem value="right">right policy</SelectItem>
                </SelectContent>
              </Select>
              <Select value={featureMode} onValueChange={(v) => setFeatureMode(v as FeatureMode)}>
                <SelectTrigger data-testid="lstd-feature-select"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="onehot">one-hot</SelectItem>
                  <SelectItem value="coordinate">coordinate</SelectItem>
                  <SelectItem value="polynomial">polynomial</SelectItem>
                  <SelectItem value="distance">distance</SelectItem>
                </SelectContent>
              </Select>
              {featureMode === 'polynomial' && (
                <SliderControl label="阶数" value={degree} min={1} max={4} step={1} onChange={setDegree} />
              )}
              <SeedControl seed={lstdSeed} onChange={setLstdSeed} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">LSTD 求解状态</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-gray-50 rounded p-2 text-center">
                  <div className="text-gray-500">状态</div>
                  <div data-testid="lstd-status" className={`font-mono font-semibold ${lstdResult.ok ? 'text-green-600' : 'text-red-600'}`}>{lstdResult.ok ? '成功' : '失败'}</div>
                </div>
                <div className="bg-gray-50 rounded p-2 text-center">
                  <div className="text-gray-500">ridge λ</div>
                  <div data-testid="lstd-ridge" className="font-mono font-semibold">{lstdResult.ok ? lstdResult.ridgeLambda : '—'}</div>
                </div>
                <div className="bg-gray-50 rounded p-2 text-center">
                  <div className="text-gray-500">minimum pivot</div>
                  <div data-testid="lstd-min-pivot" className="font-mono font-semibold">{lstdResult.minPivot.toExponential(2)}</div>
                </div>
                <div className="bg-gray-50 rounded p-2 text-center">
                  <div className="text-gray-500">condition estimate</div>
                  <div data-testid="lstd-cond" className="font-mono font-semibold">{Number.isFinite(lstdResult.conditionEstimate) ? lstdResult.conditionEstimate.toExponential(2) : '∞'}</div>
                </div>
              </div>
              {!lstdResult.ok && (
                <div className="text-red-600 bg-red-50 p-2 rounded border border-red-200">
                  求解失败：{lstdResult.reason === 'singular' ? '矩阵奇异' : lstdResult.reason === 'near-singular' ? '矩阵接近奇异（ridge 后仍无法求逆）' : '覆盖不足（A 矩阵对角线几乎为零）'}。
                </div>
              )}
            </CardContent>
          </Card>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div className="bg-gray-50 rounded p-3 border border-gray-200">
              <div className="font-semibold mb-1">LSTD w</div>
              <div className="font-mono break-all">
                [{lstdResult.ok ? lstdResult.w.map((x) => x.toFixed(3)).join(', ') : '—'}]
              </div>
            </div>
            <div className="bg-gray-50 rounded p-3 border border-gray-200">
              <div className="font-semibold mb-1">半梯度 TD(0) 最终 w</div>
              <div className="font-mono break-all">[{tdResult.weightsHistory[tdResult.weightsHistory.length - 1].map((x) => x.toFixed(3)).join(', ')}]</div>
            </div>
          </div>
          <p className="text-sm text-gray-700">
            在 on-policy、充分覆盖、相关矩阵非奇异且 TD 步长满足相应条件时，LSTD 与线性 TD 可指向同一个投影 Bellman 固定点。
          </p>
        </TabsContent>

        <TabsContent value="rls" className="mt-4 space-y-4">
          <FormulaCard
            title="Recursive Least Squares"
            formula={
              <KaTeX
                math={String.raw`P \leftarrow P - \frac{P \phi (\phi - \gamma \phi')^\top P}{1 + (\phi - \gamma \phi')^\top P \phi}, \quad b \leftarrow b + r\phi, \quad w = P b`}
                display
              />
            }
            description="每条转移用 Sherman–Morrison 恒等式递推 A⁻¹，在线得到与 LSTD 相同的最小二乘解。"
          />
          <p className="text-xs text-gray-500">
            演示使用与 LSTD 标签页相同的策略、特征与种子（共 200 回合、{rlsResult.stepsProcessed} 条转移）。
          </p>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">收敛对比（全状态 RMSE）</CardTitle>
            </CardHeader>
            <CardContent>
              <LineChart
                data={rlsChartData}
                xKey="episode"
                xLabel="回合"
                yLabel="RMSE"
                series={[
                  { key: 'rls', name: 'RLS（在线最小二乘）', color: '#8b5cf6' },
                  { key: 'td', name: '半梯度 TD(0)', color: '#2563eb' },
                  ...(lstdResult.ok
                    ? [{ key: 'lstd', name: 'LSTD 闭合解（参考）', color: '#ef4444', strokeDasharray: '6 3' }]
                    : []),
                ]}
                height={220}
              />
              <p className="mt-2 text-xs text-gray-500">
                RLS 通常远快于半梯度 TD，并在数据足够时收敛到与 LSTD 相同的解；代价是维护 P 矩阵。
              </p>
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div className="bg-gray-50 rounded p-3 border border-gray-200">
              <div className="font-semibold mb-1">RLS 最终 w</div>
              <div className="font-mono break-all">
                [{rlsResult.weightsHistory[rlsResult.weightsHistory.length - 1].map((x) => x.toFixed(3)).join(', ')}]
              </div>
            </div>
            <div className="bg-gray-50 rounded p-3 border border-gray-200">
              <div className="font-semibold mb-1">LSTD w</div>
              <div className="font-mono break-all">
                [{lstdResult.ok ? lstdResult.w.map((x) => x.toFixed(3)).join(', ') : '—'}]
              </div>
            </div>
            <div className="bg-gray-50 rounded p-3 border border-gray-200">
              <div className="font-semibold mb-1">半梯度 TD(0) 最终 w</div>
              <div className="font-mono break-all">[{tdResult.weightsHistory[tdResult.weightsHistory.length - 1].map((x) => x.toFixed(3)).join(', ')}]</div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </InteractiveDemo>
  );
}

// ------------------- 8.5 Action-value FA -------------------
function ActionValueFADemo() {
  const [algorithm, setAlgorithm] = useState<'sarsa' | 'qlearning'>('sarsa');
  const [featureMode, setFeatureMode] = useState<ActionValueFeatureMode>('shared');
  const [alpha, setAlpha] = useState(0.05);
  const [epsilon, setEpsilon] = useState(0.3);
  const [epsilonMode, setEpsilonMode] = useState<EpsilonScheduleMode>('fixed');
  const [epsilonMin, setEpsilonMin] = useState(0.01);
  const [episodes, setEpisodes] = useState(200);
  const [maxSteps, setMaxSteps] = useState(30); // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
  const [taskType, setTaskType] = useState<TaskType>('episodic');
  const [seed, setSeed] = useState(1);

  const config = configForTask(taskType);
  const qStar = useMemo(() => estimateTrueActionValues(config), [config]);

  const result = useMemo(() => {
    return actionValueFA(config, {
      alpha,
      epsilon,
      epsilonMode,
      epsilonMin,
      gamma: config.gamma,
      episodes,
      maxSteps,
      featureMode,
      algorithm,
      seed,
    });
  }, [config, algorithm, featureMode, alpha, epsilon, epsilonMode, epsilonMin, episodes, maxSteps, seed]);

  const finalQ = result.qHistory[result.qHistory.length - 1];
  const finalBehaviorPolicy = result.behaviorPolicyHistory[result.behaviorPolicyHistory.length - 1];
  const finalGreedyPolicy = result.greedyPolicyHistory[result.greedyPolicyHistory.length - 1];
  const behaviorValues = useMemo(() => policyWeightedStateValues(finalQ, finalBehaviorPolicy), [finalQ, finalBehaviorPolicy]);
  const greedyValues = useMemo(() => actionValueToStateValue(finalQ), [finalQ]);
  const finalRMSE = useMemo(() => qTableRMSE(finalQ, qStar), [finalQ, qStar]);
  const finalGreedyAgreement = useMemo(() => greedyActionAgreement(finalQ, qStar, config), [finalQ, qStar, config]);

  const chartData = useMemo(() => {
    return result.residualHistory.map((res, i) => ({
      episode: i,
      residual: res,
      rmse: qTableRMSE(result.qHistory[i], qStar),
      return: result.episodeReturnHistory[i],
      length: result.episodeLengthHistory[i],
      greedyAgreement: greedyActionAgreement(result.qHistory[i], qStar, config).agreement,
    }));
  }, [result, qStar, config]);

  const last = result.lastUpdate;

  return (
    <InteractiveDemo title={`8.5 动作值函数近似 — ${algorithm === 'sarsa' ? 'Sarsa with FA' : 'Q-learning with FA'}`}>
      <div className="grid lg:grid-cols-[1fr_360px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld
              config={config}
              policy={algorithm === 'qlearning' ? finalGreedyPolicy : finalBehaviorPolicy}
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
              yLabel={algorithm === 'qlearning' ? '最优残差 / q* RMSE / 成功率' : '行为残差 / q* RMSE / 回报'}
              series={
                algorithm === 'qlearning'
                  ? [
                      { key: 'residual', name: 'Bellman optimality residual', color: '#ef4444' },
                      { key: 'rmse', name: 'q* RMSE', color: '#2563eb' },
                      { key: 'greedyAgreement', name: '贪心动作一致率', color: '#22c55e' },
                    ]
                  : [
                      { key: 'residual', name: 'behavior-policy Bellman residual', color: '#ef4444' },
                      { key: 'rmse', name: 'q* RMSE（参考）', color: '#2563eb' },
                      { key: 'return', name: 'episode return', color: '#22c55e' },
                      { key: 'length', name: 'episode length', color: '#f59e0b' },
                    ]
              }
              height={240}
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
              <Select value={algorithm} onValueChange={(v) => setAlgorithm(v as 'sarsa' | 'qlearning')}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="sarsa">Sarsa with FA</SelectItem>
                  <SelectItem value="qlearning">Q-learning with FA</SelectItem>
                </SelectContent>
              </Select>
              <Select value={featureMode} onValueChange={(v) => setFeatureMode(v as ActionValueFeatureMode)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="onehot">one-hot(s,a) — 表格等价</SelectItem>
                  <SelectItem value="shared">共享状态特征 + 动作 one-hot + 交互</SelectItem>
                </SelectContent>
              </Select>
              <Select value={taskType} onValueChange={(v) => setTaskType(v as TaskType)}>
                <SelectTrigger data-testid="fa-actionvalue-task-select"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="episodic">episodic path-finding</SelectItem>
                  <SelectItem value="continuing">continuing</SelectItem>
                </SelectContent>
              </Select>
              <SliderControl label="学习率 α" value={alpha} min={0.001} max={0.2} step={0.001} onChange={setAlpha} />
              <SliderControl label="探索率 ε" value={epsilon} min={0} max={1} step={0.05} onChange={setEpsilon} />
              <Select value={epsilonMode} onValueChange={(v) => setEpsilonMode(v as EpsilonScheduleMode)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="fixed">fixed</SelectItem>
                  <SelectItem value="decay-floor">decay with floor</SelectItem>
                  <SelectItem value="glie">GLIE</SelectItem>
                </SelectContent>
              </Select>
              {epsilonMode !== 'fixed' && (
                <SliderControl label="最小 ε" value={epsilonMin} min={0} max={0.2} step={0.01} onChange={setEpsilonMin} />
              )}
              <SliderControl label="训练回合数" value={episodes} min={50} max={500} step={50} onChange={setEpisodes} />
              <HorizonControl value={maxSteps} onChange={setMaxSteps} />
              <SeedControl seed={seed} onChange={setSeed} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">结果</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <div>
                {algorithm === 'qlearning'
                  ? 'Q-learning 显示 Bellman optimality residual、贪心派生价值与 q* RMSE。'
                  : 'Sarsa 固定 ε 时显示 behavior-policy Bellman residual；q* RMSE 仅作为参考，不保证收敛到 q*。'}
              </div>
              <div>最终 q* RMSE：<span className="font-mono font-semibold">{finalRMSE.toFixed(4)}</span></div>
              <div>贪心动作一致率：<span className="font-mono font-semibold">{(finalGreedyAgreement.agreement * 100).toFixed(1)}%</span>（评估 {finalGreedyAgreement.evaluatedStateCount} 个非 terminal 状态）</div>
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
  const [taskType, setTaskType] = useState<TaskType>('episodic');
  const [hiddenSize, setHiddenSize] = useState(32);
  const [alpha, setAlpha] = useState(0.01);
  const [epsilon, setEpsilon] = useState(0.3);
  const [epsilonMode, setEpsilonMode] = useState<EpsilonScheduleMode>('fixed');
  const [epsilonMin, setEpsilonMin] = useState(0.01);
  const [batchSize, setBatchSize] = useState(32);
  const [replayCapacity, setReplayCapacity] = useState(2000);
  const [targetUpdateInterval, setTargetUpdateInterval] = useState(100);
  const [episodes, setEpisodes] = useState(200);
  const [maxSteps, setMaxSteps] = useState(30); // CONSISTENCY_ALLOW_DEFAULT_HORIZON: default value
  const [seed, setSeed] = useState(1);

  const config = configForTask(taskType);
  const qStar = useMemo(() => estimateTrueActionValues(config), [config]);

  const result = useMemo(() => {
    return dqnGridWorld(config, {
      hiddenSize,
      alpha,
      epsilon,
      epsilonMode,
      epsilonMin,
      gamma: config.gamma,
      batchSize,
      replayCapacity,
      targetUpdateInterval,
      episodes,
      maxSteps,
      seed,
    });
  }, [config, hiddenSize, alpha, epsilon, epsilonMode, epsilonMin, batchSize, replayCapacity, targetUpdateInterval, episodes, maxSteps, seed]);

  const lastEpisode = result.episodeHistory[result.episodeHistory.length - 1];
  const finalQ = lastEpisode?.qTableAfterEpisode ?? Array.from({ length: config.rows * config.cols }, () => new Array(5).fill(0));
  const finalPolicy = useMemo(() => greedyPolicy(finalQ), [finalQ]);
  const finalValues = useMemo(() => actionValueToStateValue(finalQ), [finalQ]);
  const finalRMSE = useMemo(() => qTableRMSE(finalQ, qStar), [finalQ, qStar]);
  const finalGreedyAgreement = useMemo(() => greedyActionAgreement(finalQ, qStar, config), [finalQ, qStar, config]);
  const finalOptimalityResidual = useMemo(() => optimalBellmanResidualQ(finalQ, config), [finalQ, config]);
  const empiricalSuccessRate = useMemo(
    () => (result.episodeHistory.length === 0 ? 0 : result.episodeHistory.filter((e) => e.success).length / result.episodeHistory.length),
    [result.episodeHistory]
  );

  const updateChartData = useMemo(
    () =>
      result.updateHistory.map((u) => ({
        step: u.update,
        loss: u.batchLoss,
        rmse: u.qRmse,
        residual: u.optimalityResidual,
        targetUpdate: u.targetSynced ? 1 : 0,
      })),
    [result.updateHistory]
  );

  const episodeChartData = useMemo(() => {
    return result.episodeHistory.map((e, i) => {
      const window = result.episodeHistory.slice(Math.max(0, i - 9), i + 1);
      const empirical = window.length === 0 ? 0 : window.filter((x) => x.success).length / window.length;
      return {
        episode: e.episode,
        cumulativeReward: e.cumulativeReward,
        episodeLength: e.episodeLength,
        empiricalSuccessRate: empirical,
        greedyAgreement: greedyActionAgreement(e.qTableAfterEpisode, qStar, config).agreement,
      };
    });
  }, [result.episodeHistory, qStar, config]);

  const targetSyncSteps = useMemo(
    () => result.updateHistory.filter((u) => u.targetSynced).map((u) => u.update),
    [result.updateHistory]
  );

  return (
    <InteractiveDemo title="8.6 Deep Q-learning：真正 mini-batch 更新">
      <div className="grid lg:grid-cols-[1fr_360px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld config={config} policy={finalPolicy} values={finalValues} showValues className="max-w-full" />
            <p className="mt-3 text-sm text-gray-500 text-center">训练结束后 DQN 的贪心策略与状态值</p>
          </div>

          <div className="bg-blue-50 rounded-xl p-4 border border-blue-100 text-sm text-gray-700">
            <h3 className="font-semibold text-blue-800 mb-2">DQN 训练流程（mini-batch gradient）</h3>
            <ol className="list-decimal pl-5 space-y-1">
              <li>与环境交互得到 transition (s, a, r, s&apos;, done)；</li>
              <li>存入经验回放 replay buffer；</li>
              <li>从 buffer 中均匀抽取 mini-batch；</li>
              <li>用<strong>同一组主网络参数</strong>计算所有样本的 prediction；</li>
              <li>用目标网络计算 y = r + γ max_a&apos; Q(s&apos;,a&apos;; w⁻)；</li>
              <li>累加各样本梯度，求平均，<strong>只执行一次参数更新</strong>；</li>
              <li>batch loss = (1/B) Σ ½(y − Q)²；</li>
              <li>每隔固定步数把主网络参数复制给目标网络。</li>
            </ol>
          </div>

          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200 space-y-6">
            <div data-testid="dqn-update-chart">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">按 training update</h4>
              <LineChart
                data={updateChartData}
                xKey="step"
                xLabel="训练更新步"
                yLabel="loss / q* RMSE / residual"
                series={[
                  { key: 'loss', name: 'batch loss', color: '#ef4444' },
                  { key: 'rmse', name: 'q* RMSE', color: '#2563eb' },
                  { key: 'residual', name: 'optimality residual', color: '#f59e0b' },
                ]}
                height={220}
              />
              {targetSyncSteps.length > 0 && (
                <div className="text-xs text-gray-500 mt-1">
                  目标网络同步步：{targetSyncSteps.join(', ')}
                </div>
              )}
            </div>
            <div data-testid="dqn-episode-chart">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">按 episode</h4>
              <LineChart
                data={episodeChartData}
                xKey="episode"
                xLabel="回合"
                yLabel="return / length / success rate"
                series={[
                  { key: 'cumulativeReward', name: 'cumulative reward', color: '#22c55e' },
                  { key: 'episodeLength', name: 'episode length', color: '#f59e0b' },
                  { key: 'empiricalSuccessRate', name: 'empirical success rate (10-ep window)', color: '#8b5cf6' },
                  { key: 'greedyAgreement', name: '贪心动作一致率', color: '#06b6d4' },
                ]}
                height={220}
              />
            </div>
          </div>

          {result.lastBatch.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <h3 className="font-semibold text-gray-800 mb-2 text-sm">
                最近一次 mini-batch（共 {result.lastBatch.length} 条，prediction 均来自更新前网络）
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
              <Select value={taskType} onValueChange={(v) => setTaskType(v as TaskType)}>
                <SelectTrigger data-testid="dqn-task-select"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="episodic">episodic path-finding</SelectItem>
                  <SelectItem value="continuing">continuing</SelectItem>
                </SelectContent>
              </Select>
              <SliderControl label="隐藏层大小" value={hiddenSize} min={8} max={64} step={8} onChange={setHiddenSize} />
              <SliderControl label="学习率 α" value={alpha} min={0.001} max={0.05} step={0.001} onChange={setAlpha} />
              <SliderControl label="探索率 ε" value={epsilon} min={0.05} max={1} step={0.05} onChange={setEpsilon} />
              <Select value={epsilonMode} onValueChange={(v) => setEpsilonMode(v as EpsilonScheduleMode)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="fixed">fixed</SelectItem>
                  <SelectItem value="decay-floor">decay with floor</SelectItem>
                  <SelectItem value="glie">GLIE</SelectItem>
                </SelectContent>
              </Select>
              {epsilonMode !== 'fixed' && (
                <SliderControl label="最小 ε" value={epsilonMin} min={0} max={0.2} step={0.01} onChange={setEpsilonMin} />
              )}
              <SliderControl label="批次大小" value={batchSize} min={8} max={64} step={8} onChange={setBatchSize} />
              <SliderControl label="回放缓冲容量" value={replayCapacity} min={500} max={5000} step={500} onChange={setReplayCapacity} />
              <SliderControl label="目标网络更新间隔" value={targetUpdateInterval} min={10} max={500} step={10} onChange={setTargetUpdateInterval} />
              <SliderControl label="训练回合数" value={episodes} min={50} max={500} step={50} onChange={setEpisodes} />
              <HorizonControl value={maxSteps} onChange={setMaxSteps} />
              <SeedControl seed={seed} onChange={setSeed} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">结果</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-1">
              <div>最终 q* RMSE：<span className="font-mono font-semibold">{finalRMSE.toFixed(4)}</span></div>
              <div>贪心动作一致率：<span className="font-mono font-semibold">{(finalGreedyAgreement.agreement * 100).toFixed(1)}%</span>（评估 {finalGreedyAgreement.evaluatedStateCount} 个状态）</div>
              <div>回合 empirical success rate：<span className="font-mono font-semibold">{(empiricalSuccessRate * 100).toFixed(1)}%</span></div>
              <div>最优残差：<span className="font-mono font-semibold">{finalOptimalityResidual.toFixed(4)}</span></div>
              <div>回放缓冲当前大小：<span className="font-mono">{result.finalReplaySize}</span></div>
              <div>训练更新步数：<span className="font-mono">{result.updateHistory.length}</span></div>
              <div>目标网络同步次数：<span className="font-mono">{targetSyncSteps.length}</span></div>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- Shared UI helpers -------------------
function SliderControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="text-sm text-gray-600 mb-1 flex justify-between">
        <span>{label}</span>
        <span className="font-mono text-gray-700">{value}</span>
      </div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={([v]) => onChange(v)} />
    </div>
  );
}

function HorizonControl({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <div>
      <label className="text-sm text-gray-600 mb-1 block">轨迹长度 H</label>
      <Select value={String(value)} onValueChange={(v) => onChange(Number(v))}>
        <SelectTrigger>
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
  );
}

function SeedControl({ seed, onChange }: { seed: number; onChange: (v: number) => void }) {
  return (
    <div className="flex gap-2 items-end">
      <div className="flex-1">
        <label className="text-sm text-gray-600 mb-1 block">随机种子</label>
        <input
          type="number"
          value={seed}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full border rounded px-2 py-1 text-sm"
        />
      </div>
      <Button onClick={() => onChange(seed + 1)} variant="outline" size="sm">
        <RefreshCw className="w-4 h-4 mr-1" />
        换种子
      </Button>
    </div>
  );
}

function SimpleBarChart({
  data,
  title,
  color,
}: {
  data: { label: string; value: number }[];
  title: string;
  color: string;
}) {
  const max = Math.max(1e-9, ...data.map((d) => Math.abs(d.value)));
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-3">
      <h4 className="text-xs font-semibold text-gray-700 mb-2">{title}</h4>
      <div className="space-y-1">
        {data.map((d) => (
          <div key={d.label} className="flex items-center gap-2 text-xs">
            <span className="w-8 text-gray-500">{d.label}</span>
            <div className="flex-1 h-3 bg-gray-100 rounded overflow-hidden">
              <div
                className="h-full rounded"
                style={{ width: `${(Math.abs(d.value) / max) * 100}%`, backgroundColor: color, opacity: d.value < 0 ? 0.5 : 1 }}
              />
            </div>
            <span className="w-16 text-right font-mono text-gray-700">{d.value.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

