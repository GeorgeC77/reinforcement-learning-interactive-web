import { Link } from 'react-router-dom';
import { BookOpen, Grid3x3, ArrowRight, ShieldAlert } from 'lucide-react';
import ConceptCard from '@/components/ConceptCard';
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
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          <ConceptCard
            icon={<Grid3x3 className="w-5 h-5" />}
            title="网格世界例子"
            description="机器人在网格中移动，目标是到达目标格并避开禁区。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="状态与动作"
            description={
              <>
                状态空间 <KaTeX math={String.raw`\mathcal{S}`} /> 与动作空间 <KaTeX math={String.raw`\mathcal{A}(s)`} />。
              </>
            }
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="状态转移"
            description={
              <>
                转移概率 <KaTeX math={String.raw`p(s'|s,a)`} /> 描述环境动态。
              </>
            }
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="奖励"
            description={
              <>
                即时奖励 <KaTeX math={String.raw`r`} /> 引导智能体行为。
              </>
            }
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="策略"
            description={
              <>
                策略 <KaTeX math={String.raw`\pi(a|s)`} /> 决定在每个状态采取什么动作。
              </>
            }
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="回报与 MDP"
            description={
              <>
                折扣回报 <KaTeX math={String.raw`G_t`} /> 与马尔可夫决策过程。
              </>
            }
          />
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
      <section className="flex justify-end">
        <Link
          to="/ch01/mdp"
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          进入马尔可夫决策过程交互演示
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
