import { useState, useMemo } from 'react';
import { Repeat, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import {
  DEFAULT_CONFIG,
  valueIteration,
  policyIteration,
} from '@/lib/rl/gridworld';

type Algorithm = 'value-iteration' | 'policy-iteration';

export default function Chapter04AlgorithmsPage() {
  const [algorithm, setAlgorithm] = useState<Algorithm>('value-iteration');
  const [step, setStep] = useState(0);

  const config = DEFAULT_CONFIG;

  const history = useMemo(() => {
    if (algorithm === 'value-iteration') {
      const { values, policies } = valueIteration(config, 200, 1e-6);
      // values[0] is the initial guess; each policy[i] corresponds to values[i+1]
      return policies.map((policy, i) => ({ values: values[i + 1], policy }));
    }
    const { values, policies } = policyIteration(config, 50, 200, 1e-6);
    // values[i] is the evaluation of policies[i]
    return values.map((v, i) => ({ values: v, policy: policies[i] }));
  }, [algorithm, config]);

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
            <Repeat className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          4.1 值迭代与策略迭代
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          两种求解贝尔曼最优方程的算法。通过逐歩观察策略和值函数，理解它们的差异。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="值迭代"
          formula={
            <KaTeX
              math={String.raw`v_{k+1}(s) = \max_a \left[ \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v_k(s') \right] = \max_a q_k(s,a)`}
              display
            />
          }
          description="每轮先对当前值函数取贪心动作，再用该贪心策略做一次贝尔曼备份。"
        />
        <FormulaCard
          title="策略迭代"
          formula={
            <KaTeX
              math={String.raw`v_{\pi_k} = (I - \gamma P_{\pi_k})^{-1} r_{\pi_k}, \quad \pi_{k+1}(s) = \arg\max_a q_{\pi_k}(s,a)`}
              display
            />
          }
          description="先完整评估当前策略，再对其做贪心改进，反复迭代。"
        />
        <FormulaCard
          title="截断策略迭代"
          formula={<KaTeX math={String.raw`\text{重复 } j \text{ 次评估备份后再做策略改进，} j=1 \Rightarrow \text{值迭代，} j=\infty \Rightarrow \text{策略迭代}`} display />}
          description="通过控制每次策略评估的步数，统一两类算法。"
        />
      </section>

      <InteractiveDemo title="逐歩对比算法">
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
              当前为第 {step} 步的{algorithm === 'value-iteration' ? '值' : '策略迭代'}结果
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">选择算法</CardTitle>
              </CardHeader>
              <CardContent className="flex gap-2">
                <Button
                  variant={algorithm === 'value-iteration' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => selectAlgorithm('value-iteration')}
                  className="flex-1"
                >
                  值迭代
                </Button>
                <Button
                  variant={algorithm === 'policy-iteration' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => selectAlgorithm('policy-iteration')}
                  className="flex-1"
                >
                  策略迭代
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">迭代控制</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between text-sm text-gray-600">
                  <span>步数</span>
                  <span className="font-mono font-semibold">{step} / {maxStep}</span>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={prev} disabled={step === 0} className="flex-1">
                    上一步
                  </Button>
                  <Button size="sm" onClick={next} disabled={step === maxStep} className="flex-1">
                    下一步
                  </Button>
                </div>
                <Button variant="outline" size="sm" onClick={reset} className="w-full">
                  重置
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">提示</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-gray-700 space-y-2">
                <p>
                  <strong>值迭代：</strong>每一步同时更新所有状态的值函数，策略在每一步都是贪心的。
                </p>
                <p>
                  <strong>策略迭代：</strong>每一步先完整评估当前策略（可跨多次 Bellman 备份），再一次性贪心改进。
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </InteractiveDemo>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">小结</h2>
        <ul className="space-y-3 text-gray-700">
          <li>值迭代把贝尔曼最优算子反复作用到值函数上，直到收敛。</li>
          <li>策略迭代在策略评估（解线性方程组或迭代）和策略改进之间交替。</li>
          <li>二者都收敛到最优值函数和最优策略；截断策略迭代是更灵活的统一框架。</li>
        </ul>
      </section>
    </div>
  );
}
