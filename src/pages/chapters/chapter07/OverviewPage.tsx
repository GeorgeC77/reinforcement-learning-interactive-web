import { Link } from 'react-router-dom';
import { BookOpen, Clock, Route, ArrowRight, ShieldAlert, FlaskConical } from 'lucide-react';
import ConceptCard from '@/components/ConceptCard';
import KaTeX from '@/components/KaTeX';
import { Ch7Playground } from '../overview-playgrounds';

export default function Chapter07OverviewPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-12 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center">
            <BookOpen className="w-9 h-9 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
          第 7 章 时序差分方法
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-6">
          本章结合蒙特卡洛的采样思想与动态规划的自举思想，学习 TD 预测与控制算法。
        </p>
        <p className="text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <Clock className="w-6 h-6 text-blue-600" />
          本章学习路线
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <ConceptCard
            icon={<Clock className="w-5 h-5" />}
            title="TD(0) 预测"
            description="用当前估计值作为自举目标，在线更新状态值。"
          />
          <ConceptCard
            icon={<Route className="w-5 h-5" />}
            title="Sarsa"
            description="同策略 TD 控制，用实际执行的下一个动作更新 q 值。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="Q-learning"
            description="异策略 TD 控制，用下一个状态的最大 q 值更新。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="n-step Sarsa"
            description="多步 TD 控制，在 TD 与蒙特卡洛回报之间做权衡。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="TD 与 MC/DP 的关系"
            description="TD 是蒙特卡洛采样与动态规划自举的折中。"
          />
          <ConceptCard
            icon={<FlaskConical className="w-5 h-5" />}
            title="补充与拓展"
            description="Expected Sarsa；TD(λ) 与 Sarsa(λ)。"
          />
        </div>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">核心公式预览</h2>
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200 space-y-3">
          <div className="text-sm text-gray-600">TD(0) 更新</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`v(s_t) \leftarrow v(s_t) + \alpha \bigl[ r_{t+1} + \gamma v(s_{t+1}) - v(s_t) \bigr]`} display />
          </div>
          <div className="text-sm text-gray-600 pt-2">Q-learning 更新</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`q(s_t,a_t) \leftarrow q(s_t,a_t) + \alpha \bigl[ r_{t+1} + \gamma \max_a q(s_{t+1},a) - q(s_t,a_t) \bigr]`} display />
          </div>
        </div>
      </section>

      {/* Interactive preview */}
      <Ch7Playground />

      <section className="flex justify-end gap-4">
        <Link
          to="/ch07/td-ext"
          className="inline-flex items-center gap-2 px-6 py-3 bg-purple-100 text-purple-700 rounded-lg font-medium hover:bg-purple-200 transition-colors"
        >
          补充与拓展
          <ArrowRight className="w-4 h-4" />
        </Link>
        <Link
          to="/ch07/td"
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          进入 TD 方法交互演示
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
