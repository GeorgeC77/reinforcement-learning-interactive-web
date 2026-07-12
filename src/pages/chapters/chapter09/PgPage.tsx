import { useState, useMemo } from 'react';
import { GitBranch, ShieldAlert, RefreshCw } from 'lucide-react';
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
import AlgorithmPlayer from '@/components/AlgorithmPlayer';
import LineChart from '@/components/LineChart';
import ConceptAccordion from '@/components/ConceptAccordion';
import {
  DEFAULT_CONFIG,
  EPISODIC_PATH_CONFIG,
  ACTION_NAMES,
  type Action,
} from '@/lib/rl/gridworld';
import {
  stateFeatures,
  featureDim,
  policyPreferences,
  policyTable,
  softmaxPolicy,
  softmaxScoreGradientFeature,
  expectedScoreZeroFeature,
  reinforceBandit,
  reinforceWithBaseline,
  reinforceMDP,
  computePolicyMetrics,
  compareBaselineVariance,
  type PGFeatureMode,
  type PGUpdateRecord,
  type MDPEpisode,
  type BanditResult,
} from '@/lib/rl/policyGradient';

type TabKey =
  | 'policy'
  | 'metrics'
  | 'discounted-pg'
  | 'average-pg'
  | 'reinforce'
  | 'baseline';

type D0Mode = 'uniform' | 'start' | 'custom';

const H_OPTIONS = [10, 20, 30, 50, 100, 200];
const FEATURE_MODES: { value: PGFeatureMode; label: string }[] = [
  { value: 'onehot', label: 'one-hot (可表示任意策略)' },
  { value: 'coordinate', label: 'coordinate [1, row, col]' },
  { value: 'distance', label: 'distance-to-target' },
];

function paramSlider(
  value: number,
  set: (v: number) => void,
  min: number,
  max: number,
  step: number,
  fixed?: number
) {
  return (
    <div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={([v]) => set(v)} />
      <div className="mt-1 text-center font-mono text-sm text-gray-700">
        {fixed !== undefined ? value.toFixed(fixed) : value}
      </div>
    </div>
  );
}

function formatVec(v: number[], fixed = 3) {
  return `[${v.map((x) => x.toFixed(fixed)).join(', ')}]`;
}

export default function Chapter09PgPage() {
  const [tab, setTab] = useState<TabKey>('policy');

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <GitBranch className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">第 9 章 策略梯度方法</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          从参数化策略表示到策略梯度定理，再到 REINFORCE 算法与基线方差降低。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="策略梯度定理（统一形式）"
          formula={
            <KaTeX
              math={String.raw`\nabla_\theta J(\theta) = \mathbb{E}_{S\sim\eta, A\sim\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(A|S) \, q_{\pi_\theta}(S,A) \right]`}
              display
            />
          }
          description="不同目标函数对应不同的状态分布 η 与动作价值 q_π。"
        />
        <FormulaCard
          title="特征化 Softmax 策略"
          formula={
            <KaTeX
              math={String.raw`h_\theta(s,a) = \theta_a^\top \phi(s), \quad \pi_\theta(a|s) = \frac{e^{h_\theta(s,a)}}{\sum_{a'} e^{h_\theta(s,a')}}`}
              display
            />
          }
          description="每个动作有一组权重 θ_a，与状态特征 φ(s) 做点积得到偏好。"
        />
        <FormulaCard
          title="Score Function"
          formula={
            <KaTeX
              math={String.raw`\nabla_{\theta_a} \log \pi(a|s) = \phi(s) \bigl( \mathbf{1}_{a=A} - \pi(a|s) \bigr)`}
              display
            />
          }
          description="softmax 的 log-策略梯度是状态特征乘以 one-hot 与策略概率之差。"
        />
      </section>

      <Tabs value={tab} onValueChange={(v) => setTab(v as TabKey)}>
        <TabsList className="grid w-full grid-cols-2 md:grid-cols-3 lg:grid-cols-6 h-auto">
          <TabsTrigger value="policy">9.1 策略函数</TabsTrigger>
          <TabsTrigger value="metrics">9.2 指标</TabsTrigger>
          <TabsTrigger value="discounted-pg">9.3 折扣 PG</TabsTrigger>
          <TabsTrigger value="average-pg">9.3 平均 PG</TabsTrigger>
          <TabsTrigger value="reinforce">9.4 REINFORCE</TabsTrigger>
          <TabsTrigger value="baseline">通往第 10 章</TabsTrigger>
        </TabsList>

        <TabsContent value="policy" className="mt-4">
          <PolicyFunctionDemo />
        </TabsContent>
        <TabsContent value="metrics" className="mt-4">
          <MetricsDemo />
        </TabsContent>
        <TabsContent value="discounted-pg" className="mt-4">
          <DiscountedPGDemo />
        </TabsContent>
        <TabsContent value="average-pg" className="mt-4">
          <AveragePGDemo />
        </TabsContent>
        <TabsContent value="reinforce" className="mt-4">
          <ReinforceDemo />
        </TabsContent>
        <TabsContent value="baseline" className="mt-4">
          <BaselineBridgeDemo />
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
                  <li>策略函数用 θ_a^T φ(s) 把状态特征映射为动作偏好，再经 softmax 得到概率。</li>
                  <li>softmax 具有常数平移不变性；score function 为 φ(s)(e_A - π(·|s))。</li>
                  <li>策略目标分折扣（fixed d0 / discounted occupancy / stationary metric）与平均奖励两种情形。</li>
                  <li>REINFORCE 用完整轨迹的折扣回报 G_t 加权 score function 更新 θ。</li>
                  <li>基线必须不依赖当前动作；它不改变期望梯度，但适当选取可降低方差。</li>
                </ul>
              ),
            },
            {
              id: 'qa1',
              title: 'Q: 为什么基线能降低方差但不改变期望？',
              content:
                '因为 Σ_a ∇log π(a|s) π(a|s) b(s) = b(s) ∇Σ_a π(a|s) = b(s) ∇1 = 0，所以减去 action-independent baseline 后期望梯度不变。',
            },
            {
              id: 'qa2',
              title: 'Q: REINFORCE 与 Actor-Critic 的主要区别？',
              content:
                'REINFORCE 用一个完整 episode 的回报 G_t 作为更新信号；Actor-Critic 用 critic 估计的 TD error，可每步更新。',
            },
          ]}
        />
      </section>
    </div>
  );
}

// ------------------- 9.1 Policy Function -------------------
function PolicyFunctionDemo() {
  const config = DEFAULT_CONFIG;
  const numStates = config.rows * config.cols;
  const [featureMode, setFeatureMode] = useState<PGFeatureMode>('coordinate');
  const [theta, setTheta] = useState<number[][]>(() => zeroThetaForMode('coordinate', config));
  const [selectedState, setSelectedState] = useState(0);
  const [selectedAction, setSelectedAction] = useState<number | null>(null);
  const [shift, setShift] = useState(0);

  // Reset theta shape when feature mode changes, keeping existing values where dimensions match.
  function handleFeatureModeChange(mode: PGFeatureMode) {
    setFeatureMode(mode);
    setTheta((prev) => {
      const fdim = featureDim(mode, config);
      return Array.from({ length: 5 }, (_, a) =>
        Array.from({ length: fdim }, (_, i) => prev[a]?.[i] ?? 0)
      );
    });
  }

  const policy = useMemo(() => policyTable(theta, featureMode, config), [theta, featureMode, config]);
  const phi = stateFeatures(selectedState, featureMode, config);
  const prefs = policyPreferences(theta, selectedState, featureMode, config);
  const probs = softmaxPolicy(prefs.map((p) => p + shift));

  function updateTheta(action: number, feature: number, value: number) {
    setTheta((prev) =>
      prev.map((row, a) => (a === action ? row.map((v, i) => (i === feature ? value : v)) : [...row]))
    );
  }

  const affectedStates = useMemo(() => {
    if (featureMode === 'onehot') {
      // In one-hot mode feature index == state index.
      return [selectedState];
    }
    const states: number[] = [];
    for (let s = 0; s < numStates; s++) {
      const f = stateFeatures(s, featureMode, config);
      if (f.some((v) => Math.abs(v) > 1e-9)) states.push(s);
    }
    return states;
  }, [selectedState, featureMode, config, numStates]);

  return (
    <InteractiveDemo title="9.1 从策略表到策略函数">
      <div className="grid lg:grid-cols-[1fr_360px] gap-6">
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">策略表 π_θ(·|s)</CardTitle>
            </CardHeader>
            <CardContent>
              <GridWorld
                config={config}
                policy={policy}
                showValues={false}
                highlightState={selectedState}
                onCellClick={setSelectedState}
                className="max-w-full"
              />
              <p className="mt-2 text-xs text-gray-600">点击格子选择要inspect的状态。</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">状态 {selectedState + 1} 的逐层计算</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-gray-50 rounded p-2">
                  <div className="text-gray-500 text-xs">φ(s)</div>
                  <div className="font-mono break-all">{formatVec(phi)}</div>
                </div>
                <div className="bg-gray-50 rounded p-2">
                  <div className="text-gray-500 text-xs">logits h_θ(s,·)</div>
                  <div className="font-mono break-all">{formatVec(prefs)}</div>
                </div>
              </div>
              <div>
                <div className="text-gray-500 text-xs mb-1">softmax π_θ(·|s)</div>
                <ProbabilityBars policy={probs} />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Softmax 常数平移不变性</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <div className="text-sm text-gray-600 mb-1">给当前状态所有 logits 加同一常数 c</div>
                {paramSlider(shift, setShift, -5, 5, 0.1, 1)}
              </div>
              <p className="text-xs text-gray-600">
                概率保持不变：π(s) = {formatVec(probs)}。这是因为分子分母同乘 e^c。
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Score Function ∇log π(a|s)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="text-sm text-gray-700">选择动作查看完整梯度矩阵（行=动作，列=特征）：</div>
              <div className="flex flex-wrap gap-2">
                {ACTION_NAMES.map((name, a) => (
                  <Button
                    key={a}
                    variant={selectedAction === a ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setSelectedAction(a)}
                  >
                    {name}
                  </Button>
                ))}
              </div>
              {selectedAction !== null && (
                <div className="text-sm">
                  <KaTeX
                    math={String.raw`\nabla_{\theta_a} \log \pi(${selectedAction}|s) = \phi(s)\bigl(\mathbf{1}_{a=${selectedAction}} - \pi(a|s)\bigr)`}
                    display
                  />
                  <div className="font-mono text-xs bg-gray-50 p-2 rounded mt-2 overflow-x-auto">
                    {softmaxScoreGradientFeature(probs, selectedAction, phi)
                      .map((row, a) => `${ACTION_NAMES[a]}: ${formatVec(row)}`)
                      .join('\n')}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">参数 θ（动作 × 特征）</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm text-gray-700 block mb-1">特征模式</label>
                <Select value={featureMode} onValueChange={(v) => handleFeatureModeChange(v as PGFeatureMode)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {FEATURE_MODES.map((m) => (
                      <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-3 max-h-[420px] overflow-y-auto pr-1">
                {theta.map((row, a) => (
                  <div key={a}>
                    <div className="text-xs font-semibold text-gray-700 mb-1">{ACTION_NAMES[a]}</div>
                    <div className="grid grid-cols-3 gap-2">
                      {row.map((v, i) => (
                        <div key={i} className="flex flex-col">
                          <span className="text-[10px] text-gray-500">{featureLabel(featureMode, i, config)}</span>
                          <input
                            type="number"
                            step={0.1}
                            value={v.toFixed(2)}
                            onChange={(e) => updateTheta(a, i, parseFloat(e.target.value) || 0)}
                            className="w-full px-1 py-1 text-center border rounded text-sm"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <Button
                variant="outline"
                size="sm"
                className="w-full"
                onClick={() => setTheta(zeroThetaForMode(featureMode, config))}
              >
                重置 θ = 0
              </Button>
              <div className="text-xs text-gray-600">
                修改 θ 后，右侧策略表会实时更新。
                {featureMode === 'onehot' ? (
                  <> 每个参数只影响对应状态。</>
                ) : (
                  <> 共享特征会影响多个状态。</>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">参数影响范围</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-sm text-gray-700 mb-2">当前特征模式会影响以下状态：</div>
              <div className="flex flex-wrap gap-1">
                {affectedStates.map((s) => (
                  <span
                    key={s}
                    className={`text-xs px-2 py-0.5 rounded ${s === selectedState ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700'}`}
                  >
                    s{s + 1}
                  </span>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

function zeroThetaForMode(mode: PGFeatureMode, config: GridWorldConfig): number[][] {
  const fdim = featureDim(mode, config);
  return Array.from({ length: 5 }, () => new Array(fdim).fill(0));
}

function featureLabel(mode: PGFeatureMode, index: number, config: GridWorldConfig): string {
  if (mode === 'onehot') return `s${index + 1}`;
  if (mode === 'coordinate') {
    return ['bias', 'row', 'col'][index] ?? `f${index}`;
  }
  return ['bias', 'row', 'col', 'dist', 'forbid'][index] ?? `f${index}`;
}

function ProbabilityBars({ policy }: { policy: number[] }) {
  return (
    <div className="space-y-2">
      {policy.map((prob, a) => (
        <div key={a} className="flex items-center gap-2 text-sm">
          <span className="w-12">{ACTION_NAMES[a]}</span>
          <div className="flex-1 h-4 bg-gray-200 rounded overflow-hidden">
            <div className="h-full bg-blue-500" style={{ width: `${prob * 100}%` }} />
          </div>
          <span className="w-14 text-right font-mono">{prob.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}

// ------------------- 9.2 Metrics -------------------
function MetricsDemo() {
  const config = DEFAULT_CONFIG;
  const numStates = config.rows * config.cols;
  const [featureMode, setFeatureMode] = useState<PGFeatureMode>('onehot');
  const [theta, setTheta] = useState<number[][]>(() => uniformThetaForMode('onehot', config));
  const [d0Mode, setD0Mode] = useState<D0Mode>('uniform');
  const [customD0, setCustomD0] = useState<number[]>(new Array(numStates).fill(1 / numStates));

  // Reset theta when feature mode changes, and apply deterministic presets only for one-hot.
  function handleFeatureModeChange(mode: PGFeatureMode) {
    setFeatureMode(mode);
    setTheta(uniformThetaForMode(mode, config));
  }

  function applyPreset(preset: 'uniform' | 'goal' | 'right') {
    setTheta(buildPresetTheta(preset, featureMode, config));
  }

  const d0 = useMemo(() => {
    if (d0Mode === 'uniform') return new Array(numStates).fill(1 / numStates);
    if (d0Mode === 'start') {
      const arr = new Array(numStates).fill(0);
      arr[config.startState] = 1;
      return arr;
    }
    const s = customD0.reduce((acc, v) => acc + v, 0);
    return s === 0 ? customD0.map(() => 1 / numStates) : customD0.map((v) => v / s);
  }, [d0Mode, customD0, numStates, config.startState]);

  const metrics = useMemo(() => computePolicyMetrics(theta, config, d0, featureMode), [theta, config, d0, featureMode]);

  const chartData = useMemo(
    () =>
      metrics.vPi.map((v, s) => ({
        state: `s${s + 1}`,
        vPi: v,
        rPi: metrics.rPi[s],
        d0: d0[s],
        dPi: metrics.dPi[s],
        contribV0: d0[s] * v,
        contribV: metrics.dPi[s] * v,
        contribR: metrics.dPi[s] * metrics.rPi[s],
      })),
    [metrics, d0]
  );

  return (
    <InteractiveDemo title="9.2 定义最优策略的指标">
      <div className="grid lg:grid-cols-[1fr_360px] gap-6">
        <div className="space-y-4">
          <div className="grid md:grid-cols-3 gap-4">
            <MetricCard title="v̄_π^0" formula="Σ_s d0(s) v_π(s)" value={metrics.Jv0} />
            <MetricCard title="v̄_π" formula="Σ_s d_π(s) v_π(s)" value={metrics.Jv} />
            <MetricCard title="r̄_π" formula="Σ_s d_π(s) r_π(s)" value={metrics.Jr} />
          </div>

          <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
            <LineChart
              data={chartData}
              xKey="state"
              xLabel="状态"
              yLabel="数值"
              series={[
                { key: 'vPi', name: 'v_π(s)', color: '#2563eb' },
                { key: 'rPi', name: 'r_π(s)', color: '#ef4444' },
                { key: 'd0', name: 'd0(s)', color: '#22c55e' },
                { key: 'dPi', name: 'd_π(s)', color: '#f59e0b' },
              ]}
              height={240}
            />
          </div>

          <div className="bg-blue-50 rounded-xl p-4 border border-blue-100 text-sm text-gray-700 space-y-2">
            <p>
              <strong>v̄_π^0</strong> 使用指定的初始分布 d0，对应固定起点的 episodic 目标。
            </p>
            <p>
              <strong>v̄_π</strong> 与 <strong>r̄_π</strong> 使用策略的 stationary distribution d_π（通过转移矩阵迭代求得，不是经验访问频率）。
            </p>
            <p>
              在折扣 stationary 指标下，教材给出近似关系
              <KaTeX math={String.raw`\bar{r}_\pi \approx (1-\gamma) \bar{v}_\pi`} display={false} />
              ；当前 v̄_π={metrics.Jv.toFixed(3)}, r̄_π={metrics.Jr.toFixed(3)}。
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">设置</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm text-gray-700 block mb-1">特征模式</label>
                <Select value={featureMode} onValueChange={(v) => handleFeatureModeChange(v as PGFeatureMode)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {FEATURE_MODES.map((m) => (
                      <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="text-sm text-gray-700 block mb-1">策略预设（仅 one-hot 精确）</label>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" onClick={() => applyPreset('uniform')}>uniform</Button>
                  <Button size="sm" variant="outline" onClick={() => applyPreset('goal')}>goal</Button>
                  <Button size="sm" variant="outline" onClick={() => applyPreset('right')}>right</Button>
                </div>
              </div>
              <div>
                <label className="text-sm text-gray-700 block mb-1">初始分布 d0</label>
                <Select value={d0Mode} onValueChange={(v) => setD0Mode(v as D0Mode)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="uniform">uniform</SelectItem>
                    <SelectItem value="start">start-state only</SelectItem>
                    <SelectItem value="custom">custom</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {d0Mode === 'custom' && (
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {customD0.map((v, s) => (
                    <div key={s} className="flex items-center gap-2 text-sm">
                      <span className="w-8">s{s + 1}</span>
                      <Slider
                        value={[v]}
                        min={0}
                        max={1}
                        step={0.05}
                        onValueChange={([nv]) => setCustomD0((prev) => prev.map((x, i) => (i === s ? nv : x)))}
                      />
                      <span className="w-10 font-mono text-right">{v.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">每状态贡献</CardTitle></CardHeader>
            <CardContent className="text-sm space-y-2 max-h-80 overflow-y-auto">
              {metrics.vPi.map((v, s) => (
                <div key={s} className="grid grid-cols-4 gap-2 text-xs">
                  <span>s{s + 1}</span>
                  <span className="font-mono">d0·v={(d0[s] * v).toFixed(2)}</span>
                  <span className="font-mono">dπ·v={(metrics.dPi[s] * v).toFixed(2)}</span>
                  <span className="font-mono">dπ·r={(metrics.dPi[s] * metrics.rPi[s]).toFixed(2)}</span>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

function uniformThetaForMode(mode: PGFeatureMode, config: GridWorldConfig): number[][] {
  return zeroThetaForMode(mode, config);
}

const GOAL_POLICY: Action[] = [1, 2, 2, 1, 2, 2, 1, 1, 4];
const RIGHT_POLICY: Action[] = [1, 1, 1, 1, 1, 1, 1, 1, 1];

function buildPresetTheta(
  preset: 'uniform' | 'goal' | 'right',
  mode: PGFeatureMode,
  config: GridWorldConfig
): number[][] {
  const numStates = config.rows * config.cols;
  if (mode !== 'onehot' || preset === 'uniform') {
    return zeroThetaForMode(mode, config);
  }
  const actions = preset === 'goal' ? GOAL_POLICY : RIGHT_POLICY;
  const bias = 5;
  return Array.from({ length: 5 }, (_, a) =>
    Array.from({ length: numStates }, (_, s) => (actions[s] === a ? bias : 0))
  );
}

function MetricCard({ title, formula, value }: { title: string; formula: string; value: number }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 text-center">
      <div className="text-sm font-semibold text-gray-800">{title}</div>
      <div className="text-xs text-gray-500 my-1">
        <KaTeX math={formula} display={false} />
      </div>
      <div className="text-2xl font-mono text-blue-600">{value.toFixed(3)}</div>
    </div>
  );
}

// ------------------- 9.3 Discounted Policy Gradient Theorem -------------------
function DiscountedPGDemo() {
  return (
    <InteractiveDemo title="9.3 折扣情形的策略梯度定理">
      <div className="space-y-4">
        <FormulaCard
          title="统一形式"
          formula={
            <KaTeX
              math={String.raw`\nabla_\theta J(\theta) = \mathbb{E}_{S\sim\eta, A\sim\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(A|S,\theta) \, q_{\pi_\theta}(S,A) \right]`}
              display
            />
          }
          description="η 是状态加权分布，q_π 是标准折扣动作值函数。"
        />

        <div className="grid md:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">A. 固定 d0 的折扣目标</CardTitle></CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <KaTeX math={String.raw`J(\theta) = \bar{v}_\pi^0 = \sum_s d_0(s) v_\pi(s)`} display />
              <KaTeX
                math={String.raw`\eta(s') = \rho_\pi(s') = \sum_s d_0(s) \sum_{k=0}^\infty \gamma^k \Pr(S_k=s'\mid S_0=s)`}
                display
              />
              <p>这是严格等式：η 是从 d0 出发的折扣占用度量。</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">B. 折扣 stationary 指标</CardTitle></CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <KaTeX math={String.raw`\bar{v}_\pi = \sum_s d_\pi(s) v_\pi(s), \quad \bar{r}_\pi = \sum_s d_\pi(s) r_\pi(s)`} display />
              <KaTeX math={String.raw`\nabla_\theta \bar{v}_\pi \approx \mathbb{E}_{S\sim d_\pi}\bigl[\cdots\bigr]`} display />
              <p>
                教材中常把 d_π 形式写成近似关系。γ 越接近 1，用 d_π 代替 ρ_π 的近似通常越准确，但它不是严格等式。
              </p>
            </CardContent>
          </Card>
        </div>

        <PGFlowDiagram />
      </div>
    </InteractiveDemo>
  );
}

// ------------------- 9.3 Average-Reward Policy Gradient -------------------
function AveragePGDemo() {
  return (
    <InteractiveDemo title="9.3 非折扣（平均奖励）情形的策略梯度定理">
      <div className="space-y-4">
        <FormulaCard
          title="平均奖励目标"
          formula={<KaTeX math={String.raw`J(\theta) = \bar{r}_\pi = \sum_s d_\pi(s) r_\pi(s)`} display />}
          description="最大化每步平均即时奖励；η 取 stationary distribution d_π。"
        />
        <FormulaCard
          title="策略梯度"
          formula={
            <KaTeX
              math={String.raw`\nabla_\theta \bar{r}_\pi = \sum_s d_\pi(s) \sum_a \nabla_\theta \pi(a|s,\theta) \, q_\pi(s,a)`}
              display
            />
          }
          description="这里的 q_π 必须使用差分动作值（differential action value）。"
        />
        <FormulaCard
          title="Differential action value"
          formula={
            <KaTeX
              math={String.raw`q_\pi(s,a) = \mathbb{E}_\pi\left[ \sum_{k=0}^\infty (R_{k+1} - \bar{r}_\pi) \,\big|\, S_0=s, A_0=a \right]`}
              display
            />
          }
          description="未折扣普通回报会发散，因此要用减去平均奖励后的差分回报。"
        />
      </div>
    </InteractiveDemo>
  );
}

function PGFlowDiagram() {
  const steps = ['d0', 'ρ_π', 'η', 'sample S', 'sample A', 'score gradient', 'q_π weight', 'policy gradient'];
  return (
    <div className="bg-gray-50 rounded-xl p-6 border border-gray-200 overflow-x-auto">
      <div className="flex items-center gap-2 min-w-max">
        {steps.map((s, i) => (
          <div key={i} className="flex items-center gap-2">
            <div className="bg-white border border-gray-300 rounded px-3 py-2 text-sm font-medium text-gray-700 shadow-sm">
              {s}
            </div>
            {i < steps.length - 1 && <span className="text-gray-400">→</span>}
          </div>
        ))}
      </div>
      <p className="text-sm text-gray-600 mt-3">
        折扣情形：从 d0 出发，通过折扣占用度量 ρ_π 得到状态权重 η，再采样 (S,A) 计算带 q_π 权重的 score gradient。
      </p>
    </div>
  );
}

// ------------------- 9.4 REINFORCE -------------------
function ReinforceDemo() {
  const [taskType, setTaskType] = useState<'episodic' | 'truncated-continuing'>('episodic');
  const config = taskType === 'episodic' ? EPISODIC_PATH_CONFIG : DEFAULT_CONFIG;
  const [featureMode, setFeatureMode] = useState<PGFeatureMode>('coordinate');
  const [alpha, setAlpha] = useState(0.05);
  const [beta, setBeta] = useState(0.1);
  const [episodes, setEpisodes] = useState(120);
  const [maxSteps, setMaxSteps] = useState(30);
  const [seed, setSeed] = useState(1);
  const [step, setStep] = useState(0);
  const [useBaseline, setUseBaseline] = useState(false);

  const result = useMemo(() => {
    return reinforceMDP(config, {
      alpha,
      beta,
      episodes,
      maxSteps,
      seed,
      useBaseline,
      featureMode,
    });
  }, [config, alpha, beta, episodes, maxSteps, seed, useBaseline, featureMode]);

  const maxStep = Math.max(0, result.updateRecords.length - 1);
  const currentRecord = result.updateRecords[Math.min(step, result.updateRecords.length - 1)];
  const currentEpisodeIndex = currentRecord ? currentRecord.episode : 0;
  const currentEpisode = result.episodes[currentEpisodeIndex];
  const currentPolicy = result.policyHistory[Math.min(currentEpisodeIndex + 1, result.policyHistory.length - 1)];

  const returnData = useMemo(
    () =>
      result.episodes.map((ep, i) => ({
        episode: i + 1,
        cumulative: ep.cumulativeReward,
        discountedG0: ep.discountedReturnG0,
        length: ep.episodeLength,
      })),
    [result]
  );

  return (
    <InteractiveDemo title="9.4 REINFORCE（教材 Algorithm 9.1）">
      <div className="space-y-6">
        <BanditWarmup />

        <div className="grid lg:grid-cols-[1fr_360px] gap-6">
          <div className="flex flex-col gap-4">
            <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
              <GridWorld config={config} policy={currentPolicy} showValues={false} className="max-w-full" />
              <p className="mt-3 text-sm text-gray-500 text-center">
                第 {currentEpisodeIndex + 1} 回合后的策略（样本 {step + 1}/{result.updateRecords.length}）
              </p>
            </div>

            <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
              <LineChart
                data={returnData}
                xKey="episode"
                xLabel="回合"
                yLabel="回报"
                series={[
                  { key: 'cumulative', name: '回合累计奖励', color: '#2563eb' },
                  { key: 'discountedG0', name: '起点折扣回报 G0', color: '#22c55e' },
                ]}
                height={200}
              />
            </div>

            {currentRecord && <PGUpdateCard record={currentRecord} episode={currentEpisode} featureMode={featureMode} />}
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-base">参数</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm text-gray-700 block mb-1">任务类型</label>
                  <Select value={taskType} onValueChange={(v) => setTaskType(v as 'episodic' | 'truncated-continuing')}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="episodic">episodic（固定起点，到达目标终止）</SelectItem>
                      <SelectItem value="truncated-continuing">truncated continuing（H 步截断，非 stationary 样本）</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="text-sm text-gray-700 block mb-1">特征模式</label>
                  <Select value={featureMode} onValueChange={(v) => setFeatureMode(v as PGFeatureMode)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {FEATURE_MODES.map((m) => (
                        <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">α</div>
                  {paramSlider(alpha, setAlpha, 0.001, 0.2, 0.001, 3)}
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">使用状态基线</span>
                  <Button variant={useBaseline ? 'default' : 'outline'} size="sm" onClick={() => setUseBaseline((b) => !b)}>
                    {useBaseline ? '开启' : '关闭'}
                  </Button>
                </div>
                {useBaseline && (
                  <div>
                    <div className="text-sm text-gray-600 mb-1">基线学习率 β</div>
                    {paramSlider(beta, setBeta, 0.001, 0.5, 0.001, 3)}
                  </div>
                )}
                <div>
                  <div className="text-sm text-gray-600 mb-1">回合数</div>
                  {paramSlider(episodes, setEpisodes, 20, 300, 10)}
                </div>
                <div>
                  <label className="text-sm text-gray-600 mb-1 block">轨迹长度 H</label>
                  <Select value={String(maxSteps)} onValueChange={(v) => setMaxSteps(Number(v))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {H_OPTIONS.map((h) => (
                        <SelectItem key={h} value={String(h)}>{h}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">随机种子</div>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(Number(e.target.value))}
                      className="flex-1 border rounded px-2 py-1 text-sm"
                    />
                    <Button onClick={() => setSeed((s) => s + 1)} variant="outline" size="sm">
                      <RefreshCw className="w-4 h-4 mr-1" />换种子
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-base">回放控制</CardTitle></CardHeader>
              <CardContent>
                <AlgorithmPlayer maxStep={maxStep} currentStep={step} onStepChange={setStep} />
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </InteractiveDemo>
  );
}

function BanditWarmup() {
  const [seed, setSeed] = useState(1);
  const [alpha, setAlpha] = useState(0.2);
  const [beta, setBeta] = useState(0.1);
  const [episodes, setEpisodes] = useState(200);

  const actionRewards = [1, 3];
  const initialTheta = [0, 0];

  const noBaseline = useMemo<BanditResult>(() => {
    return reinforceBandit(actionRewards, initialTheta, alpha, episodes, seed);
  }, [alpha, episodes, seed]);

  const withBaseline = useMemo<BanditResult>(() => {
    return reinforceWithBaseline(actionRewards, initialTheta, alpha, beta, episodes, seed);
  }, [alpha, beta, episodes, seed]);

  const noBaselineData = noBaseline.policyHistory.map((policy, i) => ({
    episode: i,
    action0: policy[0],
    action1: policy[1],
  }));

  const baselineData = withBaseline.policyHistory.map((policy, i) => ({
    episode: i,
    action0: policy[0],
    action1: policy[1],
    baseline: withBaseline.baselineHistory[i],
  }));

  return (
    <InteractiveDemo title="数学预热：无状态 Bandit">
      <div className="grid lg:grid-cols-2 gap-6">
        <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
          <div className="text-sm font-semibold text-gray-700 mb-2">REINFORCE（无基线）</div>
          <LineChart
            data={noBaselineData}
            xKey="episode"
            xLabel="回合"
            yLabel="概率"
            series={[
              { key: 'action0', name: '动作 0 (r≈1)', color: '#2563eb' },
              { key: 'action1', name: '动作 1 (r≈3)', color: '#ef4444' },
            ]}
            height={180}
          />
        </div>
        <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
          <div className="text-sm font-semibold text-gray-700 mb-2">REINFORCE + Baseline</div>
          <LineChart
            data={baselineData}
            xKey="episode"
            xLabel="回合"
            yLabel="概率 / 基线"
            series={[
              { key: 'action0', name: '动作 0', color: '#2563eb' },
              { key: 'action1', name: '动作 1', color: '#ef4444' },
              { key: 'baseline', name: '基线 b', color: '#22c55e', strokeDasharray: '4 2' },
            ]}
            height={180}
          />
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-4 items-end">
        <div className="w-40">
          <div className="text-sm text-gray-600 mb-1">α</div>
          {paramSlider(alpha, setAlpha, 0.01, 0.5, 0.01, 2)}
        </div>
        <div className="w-40">
          <div className="text-sm text-gray-600 mb-1">β</div>
          {paramSlider(beta, setBeta, 0.001, 0.5, 0.001, 3)}
        </div>
        <div className="w-40">
          <div className="text-sm text-gray-600 mb-1">回合数</div>
          {paramSlider(episodes, setEpisodes, 20, 500, 10)}
        </div>
        <div className="flex gap-2 items-end">
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            className="border rounded px-2 py-1 text-sm w-24"
          />
          <Button onClick={() => setSeed((s) => s + 1)} variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-1" />换种子
          </Button>
        </div>
      </div>
    </InteractiveDemo>
  );
}

function PGUpdateCard({
  record,
  episode,
  featureMode,
}: {
  record: PGUpdateRecord;
  episode?: MDPEpisode;
  featureMode: PGFeatureMode;
}) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 text-sm">
      <h3 className="font-semibold text-gray-800 mb-2">
        样本 (ep {record.episode + 1}, t {record.time}): s{record.state + 1} → {ACTION_NAMES[record.action]} → s{record.nextState + 1}
      </h3>
      <div className="grid md:grid-cols-2 gap-4 text-gray-700">
        <div className="space-y-1">
          <div>奖励 r：<span className="font-mono">{record.reward.toFixed(2)}</span></div>
          <div>回报 G_t：<span className="font-mono">{record.returnGt.toFixed(3)}</span></div>
          <div>behavior prob：<span className="font-mono">{record.behaviorProb.toFixed(3)}</span></div>
          <div>update prob：<span className="font-mono">{record.updateProb.toFixed(3)}</span></div>
          {record.advantage !== undefined && (
            <>
              <div>baseline before：<span className="font-mono">{record.baselineBefore?.toFixed(3)}</span></div>
              <div>advantage：<span className="font-mono">{record.advantage.toFixed(3)}</span></div>
            </>
          )}
        </div>
        <div>
          <div className="font-medium">score gradient</div>
          <div className="font-mono text-xs bg-gray-50 p-2 rounded overflow-x-auto">
            {record.scoreGradient.map((row, a) => `${ACTION_NAMES[a]}: ${formatVec(row)}`).join('\n')}
          </div>
          <div className="font-medium mt-2">parameter delta</div>
          <div className="font-mono text-xs bg-gray-50 p-2 rounded overflow-x-auto">
            {record.parameterDelta.map((row, a) => `${ACTION_NAMES[a]}: ${formatVec(row)}`).join('\n')}
          </div>
        </div>
      </div>
      {episode && (
        <div className="mt-3 text-xs text-gray-500">
          本回合 length={episode.episodeLength}，cumulative={episode.cumulativeReward.toFixed(2)}，
          G0={episode.discountedReturnG0.toFixed(2)}，success={episode.success ? '是' : '否'}，
          truncated={episode.truncated ? '是' : '否'}
        </div>
      )}
    </div>
  );
}

// ------------------- Baseline Bridge -------------------
function BaselineBridgeDemo() {
  const config = DEFAULT_CONFIG;
  const [state, setState] = useState(0);
  const [featureMode, setFeatureMode] = useState<PGFeatureMode>('coordinate');
  const [theta, setTheta] = useState<number[][]>(() => zeroThetaForMode('coordinate', config));
  const [baseline, setBaseline] = useState(1.5);
  const [numSeeds, setNumSeeds] = useState(100);

  // Keep theta shape in sync with feature mode.
  function handleFeatureModeChange(mode: PGFeatureMode) {
    setFeatureMode(mode);
    setTheta((prev) => {
      const fdim = featureDim(mode, config);
      return Array.from({ length: 5 }, (_, a) =>
        Array.from({ length: fdim }, (_, i) => prev[a]?.[i] ?? 0)
      );
    });
  }

  const phi = stateFeatures(state, featureMode, config);
  const policyState = softmaxPolicy(policyPreferences(theta, state, featureMode, config));
  const expectation = expectedScoreZeroFeature(policyState, phi);

  const banditTheta = [0.1, 0.5, -0.2, 0, 0];
  const actionRewards = [1, 3, 2, 1.5, 0.5];
  const varianceResult = useMemo(
    () => compareBaselineVariance(banditTheta, actionRewards, baseline, numSeeds),
    [baseline, numSeeds]
  );

  return (
    <InteractiveDemo title="通往第 10 章：Baseline、Advantage 与 Actor-Critic">
      <div className="space-y-4">
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-base">基线的期望为零</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-gray-700">
              对固定状态 s，枚举所有动作验证：
              <KaTeX math={String.raw`\sum_a \pi(a|s) \nabla\log\pi(a|s) \, b(s) = 0`} display={false} />
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-gray-700 block mb-1">状态</label>
                <Select value={String(state)} onValueChange={(v) => setState(Number(v))}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {Array.from({ length: config.rows * config.cols }, (_, s) => (
                      <SelectItem key={s} value={String(s)}>s{s + 1}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="text-sm text-gray-700 block mb-1">特征模式</label>
                <Select value={featureMode} onValueChange={(v) => handleFeatureModeChange(v as PGFeatureMode)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {FEATURE_MODES.map((m) => (
                      <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="font-mono text-xs bg-gray-50 p-2 rounded overflow-x-auto">
              {expectation.map((v, i) => `${featureLabel(featureMode, i, config)}: ${v.toExponential(2)}`).join(', ')}
            </div>
            <p className="text-xs text-gray-600">
              数值接近 0 即验证：不依赖当前动作的基线不改变期望梯度。注意 b(s) 不能是动作的函数。
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-base">Common-random-number 方差比较</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <div className="text-sm text-gray-600 mb-1">b（基线）</div>
                {paramSlider(baseline, setBaseline, -1, 4, 0.1, 1)}
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">种子数</div>
                {paramSlider(numSeeds, setNumSeeds, 10, 200, 10)}
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-gray-500 border-b">
                  <tr>
                    <th>分量</th>
                    <th>均值（无基线）</th>
                    <th>均值（有基线）</th>
                    <th>方差（无基线）</th>
                    <th>方差（有基线）</th>
                    <th>标准差（无基线）</th>
                    <th>标准差（有基线）</th>
                  </tr>
                </thead>
                <tbody className="text-gray-700">
                  {varianceResult.meanNoBaseline.map((_, i) => (
                    <tr key={i} className="border-b last:border-0">
                      <td>{ACTION_NAMES[i]}</td>
                      <td className="font-mono">{varianceResult.meanNoBaseline[i].toFixed(4)}</td>
                      <td className="font-mono">{varianceResult.meanBaseline[i].toFixed(4)}</td>
                      <td className="font-mono">{varianceResult.varNoBaseline[i].toFixed(4)}</td>
                      <td className="font-mono">{varianceResult.varBaseline[i].toFixed(4)}</td>
                      <td className="font-mono">{varianceResult.stdNoBaseline[i].toFixed(4)}</td>
                      <td className="font-mono">{varianceResult.stdBaseline[i].toFixed(4)}</td>
                    </tr>
                  ))}
                  <tr className="border-t-2 font-semibold">
                    <td>总梯度范数</td>
                    <td className="font-mono">{varianceResult.meanNormNoBaseline.toFixed(4)}</td>
                    <td className="font-mono">{varianceResult.meanNormBaseline.toFixed(4)}</td>
                    <td className="font-mono">{varianceResult.varNormNoBaseline.toFixed(4)}</td>
                    <td className="font-mono">{varianceResult.varNormBaseline.toFixed(4)}</td>
                    <td className="font-mono">{Math.sqrt(varianceResult.varNormNoBaseline).toFixed(4)}</td>
                    <td className="font-mono">{Math.sqrt(varianceResult.varNormBaseline).toFixed(4)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-xs text-gray-600">
              两种估计器使用相同随机数，因此均值接近。合适基线通常降低方差，但不保证每个分量都降低；任意基线不保证降低每个梯度分量的方差。
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-base">从第 9 章到第 10 章</CardTitle></CardHeader>
          <CardContent className="text-sm text-gray-700 space-y-2">
            <p>REINFORCE 是教材第 9 章的主体算法。</p>
            <p>Baseline 与 Advantage 的严格方差分析、价值函数近似 critic 等内容在第 10 章正式展开。</p>
            <p>当前页面仅说明：任何不依赖当前动作的基线都不改变期望梯度，并可能降低方差。</p>
          </CardContent>
        </Card>
      </div>
    </InteractiveDemo>
  );
}
