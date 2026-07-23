import { Link } from 'react-router-dom';
import { BookOpen, Calculator, ArrowRight, ShieldAlert, BarChart3, MousePointer2 } from 'lucide-react';
import KaTeX from '@/components/KaTeX';
import { Ch2Playground } from '../overview-playgrounds';

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
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Link
            to="/ch02/state-values"
            className="group block bg-gray-50 hover:bg-blue-50 border border-gray-200 hover:border-blue-200 rounded-xl p-5 transition-colors"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center group-hover:bg-blue-200 transition-colors">
                <BarChart3 className="w-5 h-5 text-blue-600" />
              </div>
              <h3 className="font-semibold text-gray-900">状态值函数</h3>
            </div>
            <p className="text-sm text-gray-600">
              策略评估、状态值比较，以及蒙特卡洛估计的收敛。
            </p>
          </Link>

          <Link
            to="/ch02/bellman"
            className="group block bg-gray-50 hover:bg-indigo-50 border border-gray-200 hover:border-indigo-200 rounded-xl p-5 transition-colors"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center group-hover:bg-indigo-200 transition-colors">
                <Calculator className="w-5 h-5 text-indigo-600" />
              </div>
              <h3 className="font-semibold text-gray-900">贝尔曼方程</h3>
            </div>
            <p className="text-sm text-gray-600">
              贝尔曼备份拆解、迭代求解与矩阵形式。
            </p>
          </Link>

          <Link
            to="/ch02/action-values"
            className="group block bg-gray-50 hover:bg-violet-50 border border-gray-200 hover:border-violet-200 rounded-xl p-5 transition-colors"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-violet-100 rounded-lg flex items-center justify-center group-hover:bg-violet-200 transition-colors">
                <MousePointer2 className="w-5 h-5 text-violet-600" />
              </div>
              <h3 className="font-semibold text-gray-900">动作值函数</h3>
            </div>
            <p className="text-sm text-gray-600">
              q(s,a) 与 v(s) 的关系，以及未被选择动作的 q 值。
            </p>
          </Link>

          <Link
            to="/ch02/bellman"
            className="group block bg-gray-50 hover:bg-emerald-50 border border-gray-200 hover:border-emerald-200 rounded-xl p-5 transition-colors"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-emerald-100 rounded-lg flex items-center justify-center group-hover:bg-emerald-200 transition-colors">
                <Calculator className="w-5 h-5 text-emerald-600" />
              </div>
              <h3 className="font-semibold text-gray-900">矩阵求解</h3>
            </div>
            <p className="text-sm text-gray-600">
              闭式解与迭代解的对比，已在贝尔曼方程页提供。
            </p>
          </Link>
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

      {/* Interactive preview */}
      <Ch2Playground />

      <section className="flex flex-wrap justify-end gap-3">
        <Link
          to="/ch02/state-values"
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          状态值函数
          <ArrowRight className="w-4 h-4" />
        </Link>
        <Link
          to="/ch02/bellman"
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors"
        >
          贝尔曼方程
          <ArrowRight className="w-4 h-4" />
        </Link>
        <Link
          to="/ch02/action-values"
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-violet-600 text-white rounded-lg font-medium hover:bg-violet-700 transition-colors"
        >
          动作值函数
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
