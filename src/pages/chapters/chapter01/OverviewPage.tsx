import { Link } from 'react-router-dom';
import { BookOpen, Grid3x3, ArrowRight, ShieldAlert, Route, Gift, Clock } from 'lucide-react';
import KaTeX from '@/components/KaTeX';

export default function Chapter01OverviewPage() {
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
          第 1 章 基本概念
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-6">
          强化学习的基本概念：状态、动作、状态转移、奖励、策略、回报与马尔可夫决策过程。
          全书以一个 3×3 网格世界为例贯穿始终。
        </p>
        <p className="text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      {/* Learning roadmap */}
      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <Grid3x3 className="w-6 h-6 text-blue-600" />
          本章学习路线
        </h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Link
            to="/ch01/mdp"
            className="group block bg-gray-50 hover:bg-blue-50 border border-gray-200 hover:border-blue-200 rounded-xl p-5 transition-colors"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center group-hover:bg-blue-200 transition-colors">
                <Grid3x3 className="w-5 h-5 text-blue-600" />
              </div>
              <h3 className="font-semibold text-gray-900">状态-动作-转移</h3>
            </div>
            <p className="text-sm text-gray-600">
              认识状态空间、动作空间、确定性/随机性转移 <KaTeX math={String.raw`p(s'|s,a)`} />。
            </p>
          </Link>

          <Link
            to="/ch01/policy"
            className="group block bg-gray-50 hover:bg-indigo-50 border border-gray-200 hover:border-indigo-200 rounded-xl p-5 transition-colors"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center group-hover:bg-indigo-200 transition-colors">
                <Route className="w-5 h-5 text-indigo-600" />
              </div>
              <h3 className="font-semibold text-gray-900">策略与轨迹</h3>
            </div>
            <p className="text-sm text-gray-600">
              确定性/随机性策略、策略表，以及按策略生成的轨迹与回报。
            </p>
          </Link>

          <Link
            to="/ch01/reward"
            className="group block bg-gray-50 hover:bg-rose-50 border border-gray-200 hover:border-rose-200 rounded-xl p-5 transition-colors"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-rose-100 rounded-lg flex items-center justify-center group-hover:bg-rose-200 transition-colors">
                <Gift className="w-5 h-5 text-rose-600" />
              </div>
              <h3 className="font-semibold text-gray-900">奖励设计</h3>
            </div>
            <p className="text-sm text-gray-600">
              奖励表、即时奖励陷阱，以及奖励的相对性与仿射不变性。
            </p>
          </Link>

          <Link
            to="/ch01/returns"
            className="group block bg-gray-50 hover:bg-emerald-50 border border-gray-200 hover:border-emerald-200 rounded-xl p-5 transition-colors"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-emerald-100 rounded-lg flex items-center justify-center group-hover:bg-emerald-200 transition-colors">
                <Clock className="w-5 h-5 text-emerald-600" />
              </div>
              <h3 className="font-semibold text-gray-900">回报与马尔可夫性</h3>
            </div>
            <p className="text-sm text-gray-600">
              折扣回报、回合/持续任务、无限几何级数与马尔可夫性小测验。
            </p>
          </Link>
        </div>
      </section>

      {/* Core formula preview */}
      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">核心公式预览</h2>
        <div className="space-y-6">
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <div className="text-sm text-gray-600 mb-2">折扣回报</div>
            <div className="flex justify-center">
              <KaTeX math={String.raw`G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}`} display />
            </div>
          </div>
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <div className="text-sm text-gray-600 mb-2">马尔可夫性</div>
            <div className="flex justify-center">
              <KaTeX math={String.raw`p(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, \dots) = p(s_{t+1}|s_t, a_t)`} display />
            </div>
          </div>
        </div>
      </section>

      {/* Next section CTA */}
      <section className="flex flex-wrap justify-end gap-3">
        <Link
          to="/ch01/mdp"
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          状态-动作-转移
          <ArrowRight className="w-4 h-4" />
        </Link>
        <Link
          to="/ch01/policy"
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors"
        >
          策略与轨迹
          <ArrowRight className="w-4 h-4" />
        </Link>
        <Link
          to="/ch01/reward"
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-rose-600 text-white rounded-lg font-medium hover:bg-rose-700 transition-colors"
        >
          奖励设计
          <ArrowRight className="w-4 h-4" />
        </Link>
        <Link
          to="/ch01/returns"
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-emerald-600 text-white rounded-lg font-medium hover:bg-emerald-700 transition-colors"
        >
          回报与马尔可夫性
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
