import { useState, useMemo } from 'react';
import { TrendingDown, ShieldAlert } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
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
import SeedControl from '@/components/SeedControl';
import { usePersistentState } from '@/hooks/usePersistentState';
import { mulberry32 } from '@/lib/rl/stochasticApproximation';
import {
  DEFAULT_CONFIG,
  valueIterationConvergence,
  gaussSeidelValueIteration,
  asyncValueIteration,
} from '@/lib/rl/gridworld';

type TabKey = 'contraction' | 'gauss-seidel' | 'async';

function buildConfig(gamma: number) {
  return { ...DEFAULT_CONFIG, gamma };
}

export default function Chapter04ConvergencePage() {
  const [activeTab, setActiveTab] = useState<TabKey>('contraction');
  const [gamma, setGamma] = useState(DEFAULT_CONFIG.gamma);
  const config = useMemo(() => buildConfig(gamma), [gamma]);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <TrendingDown className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          4.2 收敛性与异步动态规划
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          贝尔曼最优算子是一个压缩映射，值迭代以几何速度收敛。改变更新顺序可得到 Gauss-Seidel 与异步变体。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="贝尔曼最优算子"
          formula={
            <KaTeX
              math={String.raw`(T^* v)(s) = \max_a \left[ r(s,a) + \gamma \sum_{s'} p(s'|s,a) v(s') \right]`}
              display
            />
          }
          description="T* 把任意值函数映射到一个新的值函数，反复应用即得值迭代。"
        />
        <FormulaCard
          title="压缩映射与收敛上界"
          formula={
            <KaTeX
              math={String.raw`\|T^* v - T^* v'\|_\infty \le \gamma \|v - v'\|_\infty, \quad \|v_k - v^*\|_\infty \le \gamma^k \|v_0 - v^*\|_\infty`}
              display
            />
          }
          description="γ 越小收敛越快；γ 越大越重视长期奖励，但压缩因子越接近 1。"
        />
      </section>

      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as TabKey)}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="contraction">压缩映射</TabsTrigger>
          <TabsTrigger value="gauss-seidel">Gauss-Seidel</TabsTrigger>
          <TabsTrigger value="async">异步 DP</TabsTrigger>
        </TabsList>

        <TabsContent value="contraction" className="space-y-4 mt-4">
          <ContractionTab config={config} gamma={gamma} onGammaChange={setGamma} />
        </TabsContent>
        <TabsContent value="gauss-seidel" className="space-y-4 mt-4">
          <GaussSeidelTab config={config} />
        </TabsContent>
        <TabsContent value="async" className="space-y-4 mt-4">
          <AsyncTab config={config} />
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
                  <li>贝尔曼最优算子 T* 是 γ-压缩映射，因此值迭代有唯一不动点 v*。</li>
                  <li>误差以 γ^k 的几何速度衰减；γ 越大衰减越慢。</li>
                  <li>Gauss-Seidel 原地更新通常比 Jacobi 收敛更快，且仍保证收敛。</li>
                  <li>异步 DP 每轮只更新部分状态，只要每个状态被无限次更新，仍能收敛。</li>
                </ul>
              ),
            },
            {
              id: 'qa1',
              title: 'Q: 为什么 γ 越大收敛越慢？',
              content:
                'γ 是压缩因子。T* 把两个值函数的距离最多缩小 γ 倍，γ 越接近 1，缩小得越慢，因此需要更多迭代才能接近 v*。',
            },
            {
              id: 'qa2',
              title: 'Q: Gauss-Seidel 为什么能加速？',
              content:
                'Gauss-Seidel 在更新后面的状态时立即使用前面状态的新值，相当于每一步都在利用“更新鲜”的信息， empirical 上通常能减少迭代次数。',
            },
          ]}
        />
      </section>
    </div>
  );
}

// ------------------- Contraction Tab -------------------
function ContractionTab({
  config,
  gamma,
  onGammaChange,
}: {
  config: typeof DEFAULT_CONFIG;
  gamma: number;
  onGammaChange: (v: number) => void;
}) {
  const { errors, residuals, vStarNorm } = useMemo(() => {
    const result = valueIterationConvergence(config, 100, 1e-6);
    const vStar = valueIterationConvergence(config, 1000, 1e-12).values.at(-1) ?? [];
    const vStarNorm = Math.max(...vStar.map((x) => Math.abs(x)));
    return { errors: result.errors, residuals: result.residuals, vStarNorm };
  }, [config]);

  const data = useMemo(() => {
    return errors.map((err, i) => ({
      iteration: i + 1,
      error: err,
      residual: residuals[i] ?? null,
      bound: Math.pow(gamma, i + 1) * vStarNorm,
    }));
  }, [errors, residuals, gamma, vStarNorm]);

  return (
    <InteractiveDemo title="压缩映射：误差随迭代几何下降">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
          <LineChart
            data={data}
            xKey="iteration"
            xLabel="迭代次数 k"
            yLabel="误差 ‖v_k - v*‖∞"
            series={[
              { key: 'error', name: '实际误差', color: '#2563eb' },
              { key: 'bound', name: '理论上界 γ^k·‖v0-v*‖', color: '#ef4444', strokeDasharray: '4 4' },
              { key: 'residual', name: '相邻残差', color: '#22c55e' },
            ]}
            height={320}
          />
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">折扣因子 γ</CardTitle>
            </CardHeader>
            <CardContent>
              <Slider
                value={[gamma]}
                min={0.1}
                max={0.99}
                step={0.01}
                onValueChange={([v]) => onGammaChange(v)}
              />
              <div className="mt-2 text-center font-mono text-sm text-gray-700">
                γ = {gamma.toFixed(2)}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">观察</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <p>
                实际误差（蓝）始终位于理论几何上界（红虚线）之下，验证了压缩映射定理。
              </p>
              <p>
                相邻残差（绿）可直接在算法中计算，常用于判断收敛停止条件。
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- Gauss-Seidel Tab -------------------
function GaussSeidelTab({ config }: { config: typeof DEFAULT_CONFIG }) {
  const [step, setStep] = useState(0);

  const { jacobiData, gsHistory } = useMemo(() => {
    const vi = valueIterationConvergence(config, 100, 1e-6);
    const gs = gaussSeidelValueIteration(config, 100, 1e-6);
    const star = valueIterationConvergence(config, 1000, 1e-12).values.at(-1) ?? [];
    const jacobiErrors = vi.values.slice(1).map((v) =>
      Math.max(...v.map((val, i) => Math.abs(val - star[i])))
    );
    const gsErrors = gs.values.slice(1).map((v) =>
      Math.max(...v.map((val, i) => Math.abs(val - star[i])))
    );
    const maxLen = Math.max(jacobiErrors.length, gsErrors.length);
    const jacobiData = Array.from({ length: maxLen }, (_, i) => ({
      iteration: i + 1,
      jacobi: jacobiErrors[i] ?? null,
      gaussSeidel: gsErrors[i] ?? null,
    }));
    return { jacobiData, gsHistory: gs, vStar: star };
  }, [config]);

  const maxStep = gsHistory.values.length - 1;
  const currentValues = gsHistory.values[Math.min(step, maxStep)];
  const currentPolicy = gsHistory.policies[Math.min(step, maxStep)];

  return (
    <InteractiveDemo title="Gauss-Seidel vs Jacobi 值迭代">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld
              config={config}
              policy={currentPolicy}
              values={currentValues}
              showValues
              className="max-w-full"
            />
            <p className="mt-4 text-sm text-gray-500 text-center">
              第 {step} 次 Gauss-Seidel 扫描后的值函数与贪心策略
            </p>
          </div>
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <LineChart
              data={jacobiData}
              xKey="iteration"
              xLabel="迭代次数"
              yLabel="误差 ‖v_k - v*‖∞"
              series={[
                { key: 'jacobi', name: 'Jacobi (VI)', color: '#2563eb' },
                { key: 'gaussSeidel', name: 'Gauss-Seidel', color: '#f59e0b' },
              ]}
              height={240}
            />
          </div>
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">回放控制</CardTitle>
            </CardHeader>
            <CardContent>
              <AlgorithmPlayer maxStep={maxStep} currentStep={step} onStepChange={setStep} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">说明</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <p>
                Jacobi 值迭代先备份所有状态再统一更新；Gauss-Seidel 在扫描过程中原地写入新值。
              </p>
              <p>
                由于 3×3 网格规模很小，差异可能不明显；在更大的状态空间中 Gauss-Seidel 通常收敛更快。
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- Async DP Tab -------------------
function AsyncTab({ config }: { config: typeof DEFAULT_CONFIG }) {
  const [mode, setMode] = useState<'single-random' | 'single-sequential'>('single-random');
  const [step, setStep] = useState(0);
  const [seed, setSeed] = usePersistentState('ch04.async.seed', 1);

  const result = useMemo(() => {
    return asyncValueIteration(config, 100, 1e-6, mode, mulberry32(seed));
  }, [config, mode, seed]);

  const maxStep = result.values.length - 1;
  const currentValues = result.values[Math.min(step, maxStep)];
  const currentPolicy = result.policies[Math.min(step, maxStep)];
  const highlightState = step > 0 ? result.updatedStates[step - 1] : null;

  return (
    <InteractiveDemo title="异步动态规划：每步只更新部分状态">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
          <GridWorld
            config={config}
            policy={currentPolicy}
            values={currentValues}
            showValues
            highlightState={highlightState}
            className="max-w-full"
          />
          <p className="mt-4 text-sm text-gray-500 text-center">
            第 {step} 步：高亮显示刚刚被更新的状态
          </p>
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">更新模式</CardTitle>
            </CardHeader>
            <CardContent>
              <Select
                value={mode}
                onValueChange={(v) => {
                  setMode(v as 'single-random' | 'single-sequential');
                  setStep(0);
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder="选择更新模式" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="single-random">单状态随机</SelectItem>
                  <SelectItem value="single-sequential">单状态顺序</SelectItem>
                </SelectContent>
              </Select>
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
          {mode === 'single-random' && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">可复现性</CardTitle>
              </CardHeader>
              <CardContent>
                <SeedControl seed={seed} onChange={(v) => { setSeed(v); setStep(0); }} />
              </CardContent>
            </Card>
          )}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">说明</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <p>
                异步 DP 不要求每轮更新全部状态。只要每个状态被无限次访问，值函数仍收敛到 v*。
              </p>
              <p>
                单状态顺序类似 Gauss-Seidel；单状态随机演示更一般的异步更新。
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}
