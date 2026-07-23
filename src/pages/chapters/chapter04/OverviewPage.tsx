import { Link } from 'react-router-dom';
import { BookOpen, Repeat, Zap, ArrowRight, ShieldAlert } from 'lucide-react';
import ConceptCard from '@/components/ConceptCard';
import KaTeX from '@/components/KaTeX';
import { Ch4Playground } from '../overview-playgrounds';

export default function Chapter04OverviewPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-12 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center">
            <BookOpen className="w-9 h-9 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
          第 4 章 值迭代与策略迭代
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-6">
          本章介绍两种求解贝尔曼最优方程的经典算法：值迭代与策略迭代，以及截断策略迭代如何统一二者。
        </p>
        <p className="text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
          <Repeat className="w-6 h-6 text-blue-600" />
          本章学习路线
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <ConceptCard
            icon={<Zap className="w-5 h-5" />}
            title="值迭代"
            description="直接在值函数上做贝尔曼最优算子的重复应用。"
          />
          <ConceptCard
            icon={<Repeat className="w-5 h-5" />}
            title="策略迭代"
            description="交替进行策略评估与策略改进，直到收敛。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="截断策略迭代"
            description="只做少量策略评估就进行策略改进，统一前两种算法。"
          />
          <ConceptCard
            icon={<ArrowRight className="w-5 h-5" />}
            title="算法对比"
            description="观察两种算法在每一步的策略与值函数变化。"
          />
        </div>
      </section>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">核心公式预览</h2>
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200 space-y-3">
          <div className="text-sm text-gray-600">值迭代</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`v_{k+1}(s) = \max_a \left[ \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v_k(s') \right]`} display />
          </div>
          <div className="text-sm text-gray-600 pt-2">策略迭代（评估 + 改进）</div>
          <div className="flex justify-center">
            <KaTeX math={String.raw`v_{\pi_k} = (I - \gamma P_{\pi_k})^{-1} r_{\pi_k}, \quad \pi_{k+1}(s) = \arg\max_a q_{\pi_k}(s,a)`} display />
          </div>
        </div>
      </section>

      {/* Interactive preview */}
      <Ch4Playground />

      <section className="flex justify-end">
        <Link
          to="/ch04/vi-pi"
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          进入算法对比交互演示
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
