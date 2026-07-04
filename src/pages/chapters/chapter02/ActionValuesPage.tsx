import { useState, useMemo } from 'react';
import { MousePointer2, ShieldAlert, AlertCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import {
  DEFAULT_CONFIG,
  ACTION_NAMES,
  type Action,
  type Policy,
  deterministicPolicy,
  solveStateValues,
  computeQValues,
} from '@/lib/rl/gridworld';

export default function Chapter02ActionValuesPage() {
  const config = DEFAULT_CONFIG;
  const [policy, setPolicy] = useState<Policy>(() =>
    deterministicPolicy([1, 1, 3, 1, 1, 1, 1, 1, 4] as Action[])
  );
  const [selectedState, setSelectedState] = useState<number>(config.startState);

  const values = useMemo(() => solveStateValues(policy, config), [policy, config]);
  const qValues = useMemo(() => computeQValues(values, config), [values, config]);

  const selectedQ = qValues[selectedState];
  const selectedAction = policy[selectedState].indexOf(1);
  const stateValue = values[selectedState];
  const weightedQ = selectedQ.reduce((sum, q, a) => sum + policy[selectedState][a] * q, 0);

  function updatePolicy(state: number, action: number) {
    setPolicy((prev) =>
      prev.map((dist, s) => {
        if (s !== state) return dist;
        const d = new Array(5).fill(0);
        d[action] = 1;
        return d;
      })
    );
  }

  function qColor(q: number): string {
    const max = Math.max(...selectedQ);
    const min = Math.min(...selectedQ);
    if (max === min) return 'bg-gray-100';
    const t = (q - min) / (max - min);
    return t > 0.6 ? 'bg-emerald-100 text-emerald-800' : t < 0.4 ? 'bg-rose-100 text-rose-800' : 'bg-amber-100 text-amber-800';
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      {/* Hero */}
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-violet-100 rounded-xl flex items-center justify-center">
            <MousePointer2 className="w-8 h-8 text-violet-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          2.3 动作值函数
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          动作值 q(s,a) 表示在某状态下执行某个动作的期望回报。它是改进策略的关键。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      {/* Concept cards */}
      <section className="grid md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">定义</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            <KaTeX math={String.raw`q_\pi(s,a) = \mathbb{E}[G_t \mid S_t=s, A_t=a]`} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">与状态值的关系</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            <KaTeX math={String.raw`v_\pi(s) = \sum_a \pi(a|s) q_\pi(s,a)`} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">所有动作都有 q 值</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            即使某动作不会被当前策略选择，仍可以计算它的动作值，以发现更优动作。
          </CardContent>
        </Card>
      </section>

      {/* Interactive demo */}
      <InteractiveDemo title="动作值实验室">
        <div className="grid lg:grid-cols-[1fr_380px] gap-6">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld
              config={config}
              policy={policy}
              values={values}
              showValues
              highlightState={selectedState}
              onCellClick={setSelectedState}
              onActionClick={updatePolicy}
              editable
              className="max-w-full"
            />
            <p className="mt-4 text-sm text-gray-500 text-center">
              点击格子查看该状态所有动作的 q 值；点击箭头修改策略
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">
                  状态 s{selectedState + 1} 的动作值
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  {ACTION_NAMES.map((name, a) => {
                    const q = selectedQ[a];
                    const isSelected = policy[selectedState][a] > 0;
                    return (
                      <div
                        key={a}
                        className={`flex items-center justify-between p-3 rounded-lg border ${
                          isSelected ? 'border-violet-300 bg-violet-50' : 'border-gray-200 bg-white'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium">{name}</span>
                          {isSelected && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-violet-200 text-violet-800">
                              策略选择
                            </span>
                          )}
                        </div>
                        <span className={`font-mono font-semibold px-2 py-1 rounded ${qColor(q)}`}>
                          {q.toFixed(3)}
                        </span>
                      </div>
                    );
                  })}
                </div>

                <div className="bg-gray-50 p-3 rounded-lg border border-gray-200 text-sm text-gray-700">
                  <div className="mb-1">
                    状态值（按策略加权）：
                    <span className="font-mono font-semibold ml-1">{weightedQ.toFixed(3)}</span>
                  </div>
                  <div className="mb-1">
                    当前显示状态值：
                    <span className="font-mono font-semibold ml-1">{stateValue.toFixed(3)}</span>
                  </div>
                  <div>
                    策略选择动作：
                    <span className="font-medium ml-1">{ACTION_NAMES[selectedAction]}</span>
                  </div>
                </div>

                <div className="flex items-start gap-2 text-sm text-amber-700 bg-amber-50 p-3 rounded-lg border border-amber-100">
                  <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                  <p>
                    注意：即使某个动作不会被当前策略选择（如 a1、a4、a5），它仍然有动作值。比较这些值可以帮助我们发现更优策略。
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </InteractiveDemo>

      {/* Formula summary */}
      <section className="space-y-4">
        <FormulaCard
          title="动作值函数"
          formula={<KaTeX math={String.raw`q_\pi(s,a) = \mathbb{E}[G_t \mid S_t=s, A_t=a]`} display />}
          description="在状态 s 执行动作 a 后，再遵循策略 π 的期望折扣回报。"
        />
        <FormulaCard
          title="状态值与动作值"
          formula={<KaTeX math={String.raw`v_\pi(s) = \sum_a \pi(a|s) q_\pi(s,a)`} display />}
          description="状态值是对应动作值的策略加权平均。"
        />
        <FormulaCard
          title="动作值的贝尔曼形式"
          formula={
            <KaTeX
              math={String.raw`q_\pi(s,a) = \sum_{s',r} p(s',r|s,a) \big[ r + \gamma v_\pi(s') \big]`}
              display
            />
          }
          description="动作值由即时奖励和下一状态值组成。"
        />
      </section>
    </div>
  );
}
