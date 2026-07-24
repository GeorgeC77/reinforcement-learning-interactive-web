import { Link } from 'react-router-dom';
import { TrendingUp, ShieldCheck, ArrowRight, ShieldAlert } from 'lucide-react';
import ConceptCard from '@/components/ConceptCard';
import KaTeX from '@/components/KaTeX';

export default function Chapter12OverviewPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-12 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center">
            <ShieldCheck className="w-9 h-9 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
          第 12 章 策略优化进阶：从 REINFORCE 到 PPO
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-6">
          第 9、10 章的策略梯度每次只用一批新数据更新一次。若想让同一批数据被多次利用（提高样本效率），
          更新幅度必须受到约束——这正是 TRPO/PPO 的出发点。本章用裁剪目标函数直观解释 PPO 如何防止策略更新“步子迈太大”。
        </p>
        <p className="text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <TrendingUp className="w-6 h-6 text-blue-600" />
          本章学习路线
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <ConceptCard
            icon={<TrendingUp className="w-5 h-5" />}
            title="重要性采样比率"
            description="用旧策略的数据评估新策略：r(θ) = π_θ(a|s)/π_θ_old(a|s)。"
          />
          <ConceptCard
            icon={<ShieldCheck className="w-5 h-5" />}
            title="过大的更新为何危险"
            description="r 偏离 1 太多时，用旧数据估计的梯度不再可靠，策略可能崩溃。"
          />
          <ConceptCard
            icon={<ShieldCheck className="w-5 h-5" />}
            title="PPO 裁剪目标"
            description="min(r·A, clip(r,1-ε,1+ε)·A)：超出信任域就不再给奖励，梯度被“踩刹车”。"
          />
          <ConceptCard
            icon={<TrendingUp className="w-5 h-5" />}
            title="直观训练对比"
            description="在可控玩具问题上对比 clipped 与 unclipped 更新的稳定性。"
          />
        </div>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">核心公式预览</h2>
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200 space-y-3">
          <div className="text-sm text-gray-600">PPO 裁剪替代目标</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`L^{CLIP}(\theta) = \hat{\mathbb{E}}\left[\min\bigl(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\bigr)\right], \quad r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}`} display />
          </div>
        </div>
      </section>

      <section className="flex justify-end">
        <Link
          to="/ch12/ppo"
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          进入 PPO 直觉交互演示
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
