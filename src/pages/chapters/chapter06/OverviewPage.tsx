import { Link } from 'react-router-dom';
import { BookOpen, TrendingDown, Sigma, ArrowRight, ShieldAlert } from 'lucide-react';
import ConceptCard from '@/components/ConceptCard';
import KaTeX from '@/components/KaTeX';
import { Ch6Playground } from '../overview-playgrounds';

export default function Chapter06OverviewPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-12 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center">
            <BookOpen className="w-9 h-9 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
          第 6 章 随机逼近
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-6">
          本章介绍随机逼近的基本思想：用带有噪声的样本迭代逼近目标值，是强化学习算法的数学基础。
        </p>
        <p className="text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <TrendingDown className="w-6 h-6 text-blue-600" />
          本章学习路线
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <ConceptCard
            icon={<Sigma className="w-5 h-5" />}
            title="Robbins-Monro 算法"
            description="用带噪声的观测迭代求解方程 g(w)=0 的根。"
          />
          <ConceptCard
            icon={<TrendingDown className="w-5 h-5" />}
            title="随机梯度下降"
            description="用单个样本的梯度近似整体梯度，更新模型参数。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="收敛条件"
            description="学习率需满足 Σα_k=∞ 且 Σα_k²<∞。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="与强化学习的联系"
            description="TD、Q-learning 都是随机逼近的特例。"
          />
        </div>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">核心公式预览</h2>
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200 space-y-3">
          <div className="text-sm text-gray-600">Robbins-Monro 迭代</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`w_{k+1} = w_k - \alpha_k \tilde{g}(w_k, \eta_k)`} display />
          </div>
          <div className="text-sm text-gray-600 pt-2">随机梯度下降</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`w_{k+1} = w_k - \alpha_k \widetilde{\nabla} J(w_k)`} display />
          </div>
        </div>
      </section>

      {/* Interactive preview */}
      <Ch6Playground />

      <section className="flex justify-end">
        <Link
          to="/ch06/sa"
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          进入随机梯度下降交互演示
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
