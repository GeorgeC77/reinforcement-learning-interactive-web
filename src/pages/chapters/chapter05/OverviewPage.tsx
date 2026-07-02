import { Link } from 'react-router-dom';
import { BookOpen, Dices, TrendingUp, ArrowRight, ShieldAlert } from 'lucide-react';
import ConceptCard from '@/components/ConceptCard';
import KaTeX from '@/components/KaTeX';

export default function Chapter05OverviewPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-12 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center">
            <BookOpen className="w-9 h-9 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
          第 5 章 蒙特卡洛方法
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-6">
          本章学习如何通过与环境的交互样本估计动作值函数，并设计探索机制得到最优策略。
        </p>
        <p className="text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <Dices className="w-6 h-6 text-blue-600" />
          本章学习路线
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <ConceptCard
            icon={<TrendingUp className="w-5 h-5" />}
            title="MC Basic"
            description="用完整回合的累计回报样本平均估计动作值。"
          />
          <ConceptCard
            icon={<Dices className="w-5 h-5" />}
            title="探索起点"
            description="每个回合随机选初始状态-动作对，保证每对 (s,a) 都被访问。"
          />
          <ConceptCard
            icon={<TrendingUp className="w-5 h-5" />}
            title="ε-Greedy"
            description="以 ε 概率随机探索，否则选择当前估计最大的动作。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="策略改进"
            description="用估计的 q 值构造 ε-贪心策略，并随样本增加趋向最优。"
          />
        </div>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">核心公式预览</h2>
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200 space-y-3">
          <div className="text-sm text-gray-600">样本回报</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots`} display />
          </div>
          <div className="text-sm text-gray-600 pt-2">ε-贪心策略</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`\pi(a|s) = \begin{cases} 1-\varepsilon + \frac{\varepsilon}{|\mathcal{A}|}, & a = \arg\max_{a'} q(s,a') \\ \frac{\varepsilon}{|\mathcal{A}|}, & \text{否则} \end{cases}`} display />
          </div>
        </div>
      </section>

      <section className="flex justify-end">
        <Link
          to="/ch05/mc"
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          进入蒙特卡洛交互演示
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
