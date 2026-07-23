import { FlaskConical, ShieldAlert } from 'lucide-react';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import ConceptAccordion from '@/components/ConceptAccordion';
import TdDemo from './TdDemo';

export default function Chapter07TdExtensionsPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-purple-100 rounded-xl flex items-center justify-center">
            <FlaskConical className="w-8 h-8 text-purple-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">补充与拓展</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Expected Sarsa、TD(λ) 与 Sarsa(λ) 的延伸内容。
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

      <TdDemo
        title="补充与拓展"
        subtitle="Expected Sarsa、TD(λ) 与 Sarsa(λ) 的动态教学演示。"
        algorithms={[
          { key: 'expected', label: 'Expected Sarsa', category: 'supplement' },
          { key: 'sarsa-lambda', label: 'Sarsa(λ)', category: 'extension' },
          { key: 'td-lambda', label: 'TD(λ)', category: 'extension' },
        ]}
        defaultAlgorithm="expected"
        persistKey="ch07.td-ext"
      />

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
