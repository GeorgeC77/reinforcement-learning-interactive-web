import { Link } from 'react-router-dom';
import { Compass, Map, ArrowRight, ShieldAlert } from 'lucide-react';
import ConceptCard from '@/components/ConceptCard';
import KaTeX from '@/components/KaTeX';

export default function Chapter11OverviewPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-12 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center">
            <Compass className="w-9 h-9 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
          第 11 章 探索与规划
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-6">
          前面各章假设了探索的方式（ε-贪心）或直接给定了数据。本章把“如何探索”和“如何用模型做规划”作为主角：
          先在多臂老虎机中比较 ε-贪心、UCB 与 softmax，再用 Dyna-Q 展示模型学习如何放大真实经验的价值。
        </p>
        <p className="text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <Map className="w-6 h-6 text-blue-600" />
          本章学习路线
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <ConceptCard
            icon={<Compass className="w-5 h-5" />}
            title="多臂老虎机"
            description="只有动作与奖励的最简决策问题，是研究探索-利用权衡的试验台。"
          />
          <ConceptCard
            icon={<Compass className="w-5 h-5" />}
            title="UCB 与 softmax"
            description="不确定性驱动的乐观探索（UCB）与按价值比例探索（Boltzmann）。"
          />
          <ConceptCard
            icon={<Map className="w-5 h-5" />}
            title="模型学习"
            description="从真实转移中估计 p(s′,r|s,a)，把无模型数据变成可查询的环境。"
          />
          <ConceptCard
            icon={<Map className="w-5 h-5" />}
            title="Dyna 规划"
            description="真实经验与模型生成的模拟经验混合更新，显著提升样本效率。"
          />
        </div>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">核心公式预览</h2>
        <div className="space-y-6">
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <div className="text-sm text-gray-600 mb-2">UCB1 动作选择</div>
            <div className="flex justify-center">
              <KaTeX math={String.raw`a_t = \arg\max_a \left[ Q(a) + c\sqrt{\frac{\ln t}{N(a)}} \right]`} display />
            </div>
          </div>
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <div className="text-sm text-gray-600 mb-2">Dyna-Q 更新（真实与模拟转移同构）</div>
            <div className="flex justify-center">
              <KaTeX math={String.raw`q(s,a) \leftarrow q(s,a) + \alpha\bigl[r + \gamma \max_{a'} q(s',a') - q(s,a)\bigr]`} display />
            </div>
          </div>
        </div>
      </section>

      <section className="flex justify-end">
        <Link
          to="/ch11/exploration"
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          进入探索与规划交互演示
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
