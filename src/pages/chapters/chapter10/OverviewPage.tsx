import { Link } from 'react-router-dom';
import { BookOpen, Activity, Users, ArrowRight, ShieldAlert } from 'lucide-react';
import ConceptCard from '@/components/ConceptCard';
import KaTeX from '@/components/KaTeX';
import { Ch10Playground } from '../overview-playgrounds';

export default function Chapter10OverviewPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-12 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center">
            <BookOpen className="w-9 h-9 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
          第 10 章 Actor-Critic 方法
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-6">
          本章结合值函数近似与策略梯度：Critic 估计值函数，Actor 根据 Critic 的反馈更新策略。
        </p>
        <p className="text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <Activity className="w-6 h-6 text-blue-600" />
          本章学习路线
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <ConceptCard
            icon={<Users className="w-5 h-5" />}
            title="Actor 与 Critic"
            description="Actor 是策略网络，Critic 是值函数网络，两者协同训练。"
          />
          <ConceptCard
            icon={<Activity className="w-5 h-5" />}
            title="TD 误差驱动"
            description="用 Critic 给出的 TD 误差同时更新值函数和策略。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="优势函数"
            description="用 A(s,a)=Q(s,a)-V(s) 替代原始回报，进一步降低方差。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="现代变体"
            description="A2C、A3C、PPO、SAC 等都在 Actor-Critic 框架下发展而来。"
          />
        </div>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">核心公式预览</h2>
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200 space-y-3">
          <div className="text-sm text-gray-600">TD 误差</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`\delta_t = r_{t+1} + \gamma v(s_{t+1}) - v(s_t)`} display />
          </div>
          <div className="text-sm text-gray-600 pt-2">Actor-Critic 更新</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`w \leftarrow w + \alpha_w \delta_t \nabla_w v(s_t), \quad \theta \leftarrow \theta + \alpha_\theta \delta_t \nabla_\theta \log \pi_\theta(a_t|s_t)`} display />
          </div>
        </div>
      </section>

      {/* Interactive preview */}
      <Ch10Playground />

      <section className="flex justify-end">
        <Link
          to="/ch10/ac"
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          进入 Actor-Critic 交互演示
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
