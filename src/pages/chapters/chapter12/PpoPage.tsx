import { useMemo, useState } from 'react';
import { ShieldCheck, ShieldAlert } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import LineChart from '@/components/LineChart';
import ConceptAccordion from '@/components/ConceptAccordion';
import { usePersistentState } from '@/hooks/usePersistentState';
import { clipRatioObjective, clipObjectiveCurve, runPpoToy } from '@/lib/rl/ppo';

export default function Chapter12PpoPage() {
  const [activeTab, setActiveTab] = useState('clip');

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <ShieldCheck className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">12.1 PPO：裁剪目标函数直觉</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          想让旧数据被多次利用，就必须限制新旧策略的差异。PPO 用一行 min/clip 实现这个约束。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="概率比率"
          formula={<KaTeX math={String.raw`r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}`} display />}
          description="r=1 表示新旧策略一致；r 偏离 1 越多，用旧数据估计的新策略性能越不可信。"
        />
        <FormulaCard
          title="PPO 裁剪目标"
          formula={<KaTeX math={String.raw`L^{CLIP} = \hat{\mathbb{E}}\bigl[\min\bigl(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\bigr)\bigr]`} display />}
          description="取未裁剪与裁剪目标的较小值：一旦 r 跑出 [1-ε, 1+ε]，目标不再增加，梯度被自动截断。"
        />
      </section>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="clip">裁剪目标曲线</TabsTrigger>
          <TabsTrigger value="toy">玩具训练对比</TabsTrigger>
        </TabsList>
        <TabsContent value="clip" className="mt-4">
          <ClipCurveDemo />
        </TabsContent>
        <TabsContent value="toy" className="mt-4">
          <ToyTrainDemo />
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
                  <li>重要性采样让旧策略数据可用于新策略评估，但比率 r 偏离 1 越远估计越不可靠。</li>
                  <li>PPO 的 min/clip 目标在 r 超出 [1-ε, 1+ε] 后切断梯度，实现近似的信任域约束。</li>
                  <li>A&gt;0 时惩罚 r 过大（防止过度偏好某动作）；A&lt;0 时惩罚 r 过小（防止过度回避）。</li>
                  <li>裁剪让同一批数据可以安全地做多个 epoch 的更新，提高样本效率。</li>
                </ul>
              ),
            },
            {
              id: 'qa-min',
              title: 'Q: 为什么要取 min 而不是只 clip？',
              content:
                'clip 只在单侧截断。A>0 时 clip 挡住 r 的上侧（防止过度增加该动作概率），A<0 时挡住 r 的下侧。取 min 后，无论 A 的正负，目标函数都不会因为 r 跑出信任域而继续增大，两个方向都被约束。',
            },
            {
              id: 'qa-trpo',
              title: 'Q: 这和 TRPO 有什么关系？',
              content:
                'TRPO 用 KL 散度约束新旧策略差异，理论上更严格但实现复杂（共轭梯度 + 线搜索）。PPO 用目标函数裁剪近似同样的信任域效果，实现简单、效果相当，因此成为主流。',
            },
          ]}
        />
      </section>
    </div>
  );
}

// ------------------- Clip curve demo -------------------
function ClipCurveDemo() {
  const [epsilon, setEpsilon] = usePersistentState('ch12.clip.epsilon', 0.2);
  const [adv, setAdv] = useState(1);
  const [ratio, setRatio] = useState(1.3);

  const curve = useMemo(() => clipObjectiveCurve(adv, epsilon), [adv, epsilon]);
  const point = useMemo(() => clipRatioObjective(ratio, adv, epsilon), [ratio, adv, epsilon]);

  const chartData = useMemo(
    () =>
      curve.map((d) => ({
        ...d,
        point: Math.abs(d.r - ratio) < 0.02 ? d.clipped : (undefined as unknown as number),
      })),
    [curve, ratio]
  );

  return (
    <InteractiveDemo title="裁剪目标函数：min(r·A, clip(r)·A) 的形状">
      <div className="grid lg:grid-cols-[1fr_320px] gap-6">
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">A = {adv > 0 ? '+1' : '−1'}，ε = {epsilon.toFixed(2)}</CardTitle>
            </CardHeader>
            <CardContent>
              <LineChart
                data={chartData}
                xKey="r"
                xLabel="比率 r(θ)"
                yLabel="目标值"
                series={[
                  { key: 'unclipped', name: 'r·A（未裁剪）', color: '#94a3b8', strokeDasharray: '6 3' },
                  { key: 'clipped', name: 'min(r·A, clip(r)·A)', color: '#2563eb' },
                ]}
                height={240}
              />
              <p className="mt-2 text-xs text-gray-500">
                蓝线在 r∈[1-ε, 1+ε] 内与灰线重合；跑出该区间的部分被“削平”，梯度变为 0。
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">当前点 r = {ratio.toFixed(2)}</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-3 gap-3 text-center text-sm">
              <div className="bg-gray-50 rounded p-2 border border-gray-200">
                <div className="text-xs text-gray-500">r·A</div>
                <div className="font-mono font-semibold">{point.unclipped.toFixed(3)}</div>
              </div>
              <div className="bg-gray-50 rounded p-2 border border-gray-200">
                <div className="text-xs text-gray-500">clip(r)·A</div>
                <div className="font-mono font-semibold">{(point.ratioClipped * adv).toFixed(3)}</div>
              </div>
              <div className="bg-blue-50 rounded p-2 border border-blue-200">
                <div className="text-xs text-gray-500">min（采用）</div>
                <div className="font-mono font-semibold text-blue-700">{point.clipped.toFixed(3)}</div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1"><span>clip 范围 ε</span><span className="font-mono">{epsilon.toFixed(2)}</span></div>
                <Slider value={[epsilon]} min={0.05} max={0.5} step={0.05} onValueChange={([v]) => setEpsilon(v)} />
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1"><span>优势 A</span><span className="font-mono">{adv > 0 ? '+1' : '−1'}</span></div>
                <Slider value={[adv]} min={-1} max={1} step={2} onValueChange={([v]) => setAdv(v)} />
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1"><span>比率 r</span><span className="font-mono">{ratio.toFixed(2)}</span></div>
                <Slider value={[ratio]} min={0} max={2} step={0.05} onValueChange={([v]) => setRatio(v)} />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">读图指南</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <p>A&gt;0（好动作）：r 越大目标越大，但 r&gt;1+ε 后不再增长——防止把该动作概率推得过高。</p>
              <p>A&lt;0（坏动作）：r 越小目标越大，但 r&lt;1-ε 后不再增长——防止把该动作概率压得过低。</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- Toy training demo -------------------
function ToyTrainDemo() {
  const [alpha, setAlpha] = usePersistentState('ch12.toy.alpha', 1.0);
  const [epsilon, setEpsilon] = usePersistentState('ch12.toy.epsilon', 0.2);
  const [mu1, setMu1] = usePersistentState('ch12.toy.mu1', 1.0);

  const clipped = useMemo(
    () => runPpoToy({ mu0: 0, mu1, alpha, epsilon, iterations: 80, mode: 'clipped' }),
    [mu1, alpha, epsilon]
  );
  const unclipped = useMemo(
    () => runPpoToy({ mu0: 0, mu1, alpha, epsilon, iterations: 80, mode: 'unclipped' }),
    [mu1, alpha, epsilon]
  );

  const chartData = useMemo(
    () =>
      clipped.map((s, i) => ({
        iteration: s.iteration,
        clipped: s.p1,
        unclipped: unclipped[i].p1,
      })),
    [clipped, unclipped]
  );

  return (
    <InteractiveDemo title="玩具问题：单状态两动作，clipped vs unclipped 更新">
      <div className="grid lg:grid-cols-[1fr_320px] gap-6">
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">P(选择更优动作 a₁) 随更新的变化</CardTitle>
            </CardHeader>
            <CardContent>
              <LineChart
                data={chartData}
                xKey="iteration"
                xLabel="更新次数"
                yLabel="p₁ = π(a₁)"
                series={[
                  { key: 'clipped', name: 'PPO clip', color: '#2563eb' },
                  { key: 'unclipped', name: '未裁剪 r·A', color: '#ef4444', strokeDasharray: '6 3' },
                ]}
                height={240}
              />
              <p className="mt-2 text-xs text-gray-500">
                两者都会学到偏好 a₁（均值 {mu1.toFixed(1)} vs 0），但裁剪版每一步的比率都被限制在信任域内，曲线更平滑；
                未裁剪版在大学习率下可能一步冲过头。
              </p>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1"><span>学习率 α</span><span className="font-mono">{alpha.toFixed(2)}</span></div>
                <Slider value={[alpha]} min={0.1} max={3} step={0.1} onValueChange={([v]) => setAlpha(v)} />
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1"><span>clip 范围 ε</span><span className="font-mono">{epsilon.toFixed(2)}</span></div>
                <Slider value={[epsilon]} min={0.05} max={0.5} step={0.05} onValueChange={([v]) => setEpsilon(v)} />
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1"><span>a₁ 均值 μ₁</span><span className="font-mono">{mu1.toFixed(1)}</span></div>
                <Slider value={[mu1]} min={0.2} max={3} step={0.2} onValueChange={([v]) => setMu1(v)} />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">读图指南</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <p>把 α 调大（如 2.5–3）：未裁剪曲线会振荡或冲顶，裁剪曲线仍然受控。</p>
              <p>把 ε 调小：信任域更窄，裁剪曲线更保守、更慢但更稳。</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}
