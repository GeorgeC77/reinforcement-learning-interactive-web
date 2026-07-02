import { Link } from 'react-router-dom';
import { BookOpen, Brain, Layers, ArrowRight, ShieldAlert } from 'lucide-react';
import ConceptCard from '@/components/ConceptCard';
import KaTeX from '@/components/KaTeX';

export default function Chapter08OverviewPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-12 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center">
            <BookOpen className="w-9 h-9 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
          第 8 章 值函数方法
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-6">
          本章学习用参数化函数近似值函数，将强化学习扩展到连续状态空间和大规模问题。
        </p>
        <p className="text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <Brain className="w-6 h-6 text-blue-600" />
          本章学习路线
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <ConceptCard
            icon={<Layers className="w-5 h-5" />}
            title="线性值函数近似"
            description="用人工特征 φ(s) 的线性组合逼近状态值。"
          />
          <ConceptCard
            icon={<Brain className="w-5 h-5" />}
            title="梯度下降更新"
            description="最小化预测值与 TD 目标之间的均方误差。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="特征工程"
            description="多项式、傅里叶、径向基等特征对近似能力的影响。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="神经网络近似"
            description="用深度神经网络代替线性模型，得到 DQN 等算法。"
          />
        </div>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">核心公式预览</h2>
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200 space-y-3">
          <div className="text-sm text-gray-600">线性近似</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`\hat{v}(s, w) = \phi(s)^\top w = \sum_i \phi_i(s) w_i`} display />
          </div>
          <div className="text-sm text-gray-600 pt-2">梯度 TD 更新</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`w \leftarrow w + \alpha \bigl[ U_t - \hat{v}(s_t, w) \bigr] \nabla_w \hat{v}(s_t, w)`} display />
          </div>
        </div>
      </section>

      <section className="flex justify-end">
        <Link
          to="/ch08/fa"
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          进入值函数近似交互演示
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
