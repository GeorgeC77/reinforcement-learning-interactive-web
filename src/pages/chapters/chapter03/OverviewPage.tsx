import { Link } from 'react-router-dom';
import { BookOpen, Trophy, ArrowRight, ShieldAlert } from 'lucide-react';
import ConceptCard from '@/components/ConceptCard';
import KaTeX from '@/components/KaTeX';
import { Ch3Playground } from '../overview-playgrounds';

export default function Chapter03OverviewPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-12 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center">
            <BookOpen className="w-9 h-9 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
          第 3 章 最优状态值与贝尔曼最优方程
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-6">
          本章定义最优策略与最优状态值，并引入求解它们的核心工具——贝尔曼最优方程（BOE）。
        </p>
        <p className="text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <Trophy className="w-6 h-6 text-blue-600" />
          本章学习路线
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <ConceptCard
            icon={<Trophy className="w-5 h-5" />}
            title="最优策略"
            description="若一个策略在所有状态上的值都不低于任何其他策略，则它是最优策略。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="贝尔曼最优方程"
            description={
              <>
                <KaTeX math={String.raw`v(s) = \max_\pi \sum_a \pi(a|s) q(s,a)`} /> 刻画最优值函数。
              </>
            }
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="压缩映射"
            description="BOE 右端是压缩映射，保证唯一解与迭代收敛。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="影响最优策略的因素"
            description="折扣因子 γ、奖励设计会改变最优策略。"
          />
        </div>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">核心公式预览</h2>
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
          <div className="text-sm text-gray-600 mb-2">贝尔曼最优方程</div>
          <div className="flex justify-center">
            <KaTeX
              math={String.raw`v(s) = \max_\pi \sum_a \pi(a|s) \left[ \sum_r p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v(s') \right] = \max_\pi \sum_a \pi(a|s) q(s,a)`}
              display
            />
          </div>
        </div>
      </section>

      {/* Interactive preview */}
      <Ch3Playground />

      <section className="flex justify-end">
        <Link
          to="/ch03/boe"
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          进入贝尔曼最优方程交互演示
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
