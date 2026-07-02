import { Link } from 'react-router-dom';
import { BookOpen, Calculator, ArrowRight, ShieldAlert } from 'lucide-react';
import ConceptCard from '@/components/ConceptCard';
import KaTeX from '@/components/KaTeX';

export default function Chapter02OverviewPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      {/* Hero */}
      <section className="text-center py-12 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center">
            <BookOpen className="w-9 h-9 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
          第 2 章 状态值与贝尔曼方程
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-6">
          状态值函数用于评估一个策略的好坏；贝尔曼方程则是求解状态值的强大工具。
        </p>
        <p className="text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      {/* Learning roadmap */}
      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <Calculator className="w-6 h-6 text-blue-600" />
          本章学习路线
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <ConceptCard
            icon={<Calculator className="w-5 h-5" />}
            title="状态值函数"
            description={
              <>
                <KaTeX math={String.raw`v_\pi(s) = \mathbb{E}[G_t | S_t = s]`} />：从状态 <KaTeX math={String.raw`s`} /> 出发，遵循策略 <KaTeX math={String.raw`\pi`} /> 的期望回报。
              </>
            }
          />
          <ConceptCard
            icon={<Calculator className="w-5 h-5" />}
            title="贝尔曼方程"
            description={
              <>
                值函数之间的自举关系：<KaTeX math={String.raw`v = r_\pi + \gamma P_\pi v`} />。
              </>
            }
          />
          <ConceptCard
            icon={<Calculator className="w-5 h-5" />}
            title="矩阵形式"
            description={
              <>
                闭式解 <KaTeX math={String.raw`v = (I - \gamma P_\pi)^{-1} r_\pi`} />。
              </>
            }
          />
          <ConceptCard
            icon={<Calculator className="w-5 h-5" />}
            title="迭代求解"
            description={
              <>
                <KaTeX math={String.raw`v_{k+1} = r_\pi + \gamma P_\pi v_k`} /> 的收敛过程。
              </>
            }
          />
        </div>
      </section>

      {/* Core formula preview */}
      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">核心公式预览</h2>
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
          <div className="text-sm text-gray-600 mb-2">贝尔曼方程（元素形式）</div>
          <div className="flex justify-center">
            <KaTeX
              math={String.raw`v_\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a) \big[r + \gamma v_\pi(s')\big]`}
              display
            />
          </div>
        </div>
      </section>

      <section className="flex justify-end">
        <Link
          to="/ch02/bellman"
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          进入贝尔曼方程交互演示
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
