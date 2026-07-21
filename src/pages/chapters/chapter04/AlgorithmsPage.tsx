import { useState, useMemo, useEffect } from 'react';
import { Repeat, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import AlgorithmPlayer from '@/components/AlgorithmPlayer';
import LineChart from '@/components/LineChart';
import ConceptAccordion from '@/components/ConceptAccordion';
import {
  DEFAULT_CONFIG,
  valueIteration,
  policyIteration,
  iterateStateValues,
  greedyPolicy,
  computeQValues,
  randomPolicy,
} from '@/lib/rl/gridworld';

type Algorithm = 'vi' | 'pi' | 'tpi';

function truncatedPolicyIteration(
  config: typeof DEFAULT_CONFIG,
  evalSteps: number,
  maxIterations: number = 100
) {
  const numStates = config.rows * config.cols;
  let policy = randomPolicy(numStates);
  const values: number[][] = [];
  const policies: typeof policy[] = [policy];

  for (let k = 0; k < maxIterations; k++) {
    const vHistory = iterateStateValues(policy, config, evalSteps, null);
    const v = vHistory[vHistory.length - 1];
    values.push(v);

    const q = computeQValues(v, config);
    const newPolicy = greedyPolicy(q);

    const isSame = policy.every((dist, s) =>
      dist.every((p, a) => Math.abs(p - newPolicy[s][a]) < 1e-9)
    );

    policy = newPolicy;
    policies.push(policy);
    if (isSame) break;
  }

  return { values, policies };
}

export default function Chapter04AlgorithmsPage() {
  const [algorithm, setAlgorithm] = useState<Algorithm>('vi');
  const [step, setStep] = useState(0);
  const [tpiEvalSteps, setTpiEvalSteps] = useState(5);

  const config = DEFAULT_CONFIG;

  useEffect(() => {
    setStep(0);
  }, [algorithm, tpiEvalSteps]);

  const history = useMemo(() => {
    if (algorithm === 'vi') {
      const { values, policies } = valueIteration(config, 200, 1e-6);
      return policies.map((policy, i) => ({ values: values[i + 1], policy }));
    }
    if (algorithm === 'pi') {
      const { values, policies } = policyIteration(config, 50, 200, 1e-6);
      return values.map((v, i) => ({ values: v, policy: policies[i] }));
    }
    const { values, policies } = truncatedPolicyIteration(config, tpiEvalSteps);
    return values.map((v, i) => ({ values: v, policy: policies[i] }));
  }, [algorithm, config, tpiEvalSteps]);

  const current = history[Math.min(step, history.length - 1)];
  const maxStep = history.length - 1;

  const convergenceData = useMemo(() => {
    const vi = valueIteration(config, 200, 1e-6).values;
    const pi = policyIteration(config, 50, 200, 1e-6).values;
    const tpi = truncatedPolicyIteration(config, 5).values;
    const maxLen = Math.max(vi.length, pi.length, tpi.length);
    return Array.from({ length: maxLen }, (_, i) => ({
      iteration: i,
      vi: vi[i]?.[0] ?? null,
      pi: pi[i]?.[0] ?? null,
      tpi: tpi[i]?.[0] ?? null,
    }));
  }, [config]);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Repeat className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          4.1 值迭代、策略迭代与截断策略迭代
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          三种求解贝尔曼最优方程的动态规划算法：值迭代、策略迭代，以及统一二者的截断策略迭代。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <Tabs defaultValue="vi" onValueChange={(v) => setAlgorithm(v as Algorithm)}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="vi">值迭代 (VI)</TabsTrigger>
          <TabsTrigger value="pi">策略迭代 (PI)</TabsTrigger>
          <TabsTrigger value="tpi">截断策略迭代 (TPI)</TabsTrigger>
        </TabsList>

        <TabsContent value="vi" className="space-y-4 mt-4">
          <FormulaCard
            title="值迭代"
            formula={
              <KaTeX
                math={String.raw`v_{k+1}(s) = \max_a \left[ \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v_k(s') \right] = \max_a q_k(s,a)`}
                display
              />
            }
            description="每一步直接对当前值函数应用贝尔曼最优算子。策略是隐式的——始终对当前 q 值取贪心。"
          />
        </TabsContent>

        <TabsContent value="pi" className="space-y-4 mt-4">
          <FormulaCard
            title="策略迭代"
            formula={
              <KaTeX
                math={String.raw`v_{\pi_k} = (I - \gamma P_{\pi_k})^{-1} r_{\pi_k}, \quad \pi_{k+1}(s) = \arg\max_a q_{\pi_k}(s,a)`}
                display
              />
            }
            description="两个子步骤交替：先完全评估当前策略（PE），再贪心改进（PI）。"
          />
        </TabsContent>

        <TabsContent value="tpi" className="space-y-4 mt-4">
          <FormulaCard
            title="截断策略迭代"
            formula={
              <KaTeX
                math={String.raw`v_{k+1} = T_{\pi_k}^{j}(v_k), \quad \pi_{k+1} = \mathcal{G}(v_{k+1})`}
                display
              />
            }
            description="每次策略评估只做 j 步 Bellman 备份就做策略改进。j=1 退化为值迭代，j=∞ 退化为策略迭代。"
          />
        </TabsContent>
      </Tabs>

      <InteractiveDemo title="算法逐歩回放与对比">
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
              第 {step} 次外迭代后的策略与值函数
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">算法选择</CardTitle>
              </CardHeader>
              <CardContent className="flex gap-2">
                {[
                  { key: 'vi', label: '值迭代' },
                  { key: 'pi', label: '策略迭代' },
                  { key: 'tpi', label: '截断 PI' },
                ].map(({ key, label }) => (
                  <Button
                    key={key}
                    variant={algorithm === key ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setAlgorithm(key as Algorithm)}
                    className="flex-1"
                  >
                    {label}
                  </Button>
                ))}
              </CardContent>
            </Card>

            {algorithm === 'tpi' && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">评估步数 j</CardTitle>
                </CardHeader>
                <CardContent>
                  <Slider
                    value={[tpiEvalSteps]}
                    min={1}
                    max={30}
                    step={1}
                    onValueChange={([v]) => setTpiEvalSteps(v)}
                  />
                  <div className="mt-2 text-center font-mono text-sm text-gray-700">
                    j = {tpiEvalSteps}
                  </div>
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">回放控制</CardTitle>
              </CardHeader>
              <CardContent>
                <AlgorithmPlayer
                  maxStep={maxStep}
                  currentStep={step}
                  onStepChange={setStep}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">提示</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-gray-700 space-y-2">
                <p><strong>VI：</strong>每次外迭代只做 1 次值备份 + 1 次贪心。</p>
                <p><strong>PI：</strong>每次外迭代做完整策略评估（多次备份）+ 1 次贪心。</p>
                <p><strong>TPI：</strong>调节 j，观察 VI 与 PI 只是两种极端。</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </InteractiveDemo>

      <InteractiveDemo title="收敛曲线对比">
        <LineChart
          data={convergenceData}
          xKey="iteration"
          xLabel="外迭代次数"
          yLabel="v(s_1)"
          series={[
            { key: 'vi', name: 'Value Iteration', color: '#2563eb' },
            { key: 'pi', name: 'Policy Iteration', color: '#ef4444' },
            { key: 'tpi', name: 'Truncated PI (j=5)', color: '#22c55e' },
          ]}
        />
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
                  <li>值迭代直接反复应用贝尔曼最优算子，隐式维护贪心策略。</li>
                  <li>策略迭代在策略评估与策略改进之间交替，收敛通常更快。</li>
                  <li>截断策略迭代通过控制评估步数 j 统一二者。</li>
                  <li>三者都是动态规划算法，都需要模型。</li>
                </ul>
              ),
            },
            {
              id: 'qa1',
              title: 'Q: 值迭代和策略迭代哪个更好？',
              content:
                '没有绝对答案。值迭代每步计算量小但外迭代次数多；策略迭代外迭代次数少但每次评估计算量大。截断策略迭代是更实用的折中。',
            },
            {
              id: 'qa2',
              title: 'Q: 为什么截断策略迭代也能收敛？',
              content:
                '只要策略评估足够接近真实值，贪心改进就能保证策略不退化。TPI 的评估虽不完整，但仍满足这一条件。',
            },
          ]}
        />
      </section>
    </div>
  );
}
