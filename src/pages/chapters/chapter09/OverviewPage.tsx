import { Link } from 'react-router-dom';
import { BookOpen, GitBranch, Target, ArrowRight, ShieldAlert } from 'lucide-react';
import ConceptCard from '@/components/ConceptCard';
import KaTeX from '@/components/KaTeX';
import { Ch9Playground } from '../overview-playgrounds';

export default function Chapter09OverviewPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-12 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center">
            <BookOpen className="w-9 h-9 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
          第 9 章 策略梯度方法
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-6">
          本章直接参数化策略，用梯度上升优化策略目标，是连续动作空间和大规模离散动作空间的重要方法。
        </p>
        <p className="text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <GitBranch className="w-6 h-6 text-blue-600" />
          本章学习路线
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <ConceptCard
            icon={<Target className="w-5 h-5" />}
            title="策略梯度定理"
            description="直接对策略目标求梯度，得到期望形式的策略梯度。"
          />
          <ConceptCard
            icon={<GitBranch className="w-5 h-5" />}
            title="REINFORCE"
            description="用蒙特卡洛回报估计策略梯度，进行策略更新。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="Baseline"
            description="引入基线降低梯度估计方差，加速收敛。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="Softmax 策略"
            description="离散动作常用参数化方式，概率由动作偏好经 softmax 得到。"
          />
        </div>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">核心公式预览</h2>
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200 space-y-3">
          <div className="text-sm text-gray-600">策略梯度定理</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot q_{\pi_\theta}(s,a) \right]`} display />
          </div>
          <div className="text-sm text-gray-600 pt-2">REINFORCE 更新</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`\theta \leftarrow \theta + \alpha \cdot G_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)`} display />
          </div>
        </div>
      </section>

      {/* Interactive preview */}
      <Ch9Playground />

      <section className="flex justify-end">
        <Link
          to="/ch09/pg"
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          进入 REINFORCE 交互演示
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
