import { useState, useMemo } from 'react';
import { Trophy, RotateCcw, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import {
  DEFAULT_CONFIG,
  valueIteration,
} from '@/lib/rl/gridworld';

export default function Chapter03BoePage() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);

  const { values, policies } = useMemo(() => {
    return valueIteration(config, 200, 1e-6);
  }, [config]);

  const optimalValues = values[values.length - 1];
  const optimalPolicy = policies[policies.length - 1];
  const iterations = values.length - 1;

  function reset() {
    setConfig(DEFAULT_CONFIG);
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Trophy className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          3.1 贝尔曼最优方程
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          贝尔曼最优方程是求解最优值函数与最优策略的工具。通过调节折扣因子与奖励，观察最优策略如何变化。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="贝尔曼最优方程"
          formula={
            <KaTeX
              math={String.raw`v(s) = \max_\pi \sum_a \pi(a|s) q(s,a), \quad q(s,a) = \sum_r p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v(s')`}
              display
            />
          }
          description="最优值函数等于在所有策略中选取使期望动作值最大的策略。"
        />
        <FormulaCard
          title="贪心最优策略"
          formula={<KaTeX math={String.raw`\pi^*(a|s) = \begin{cases} 1, & a = \arg\max_a q^*(s,a) \\ 0, & \text{否则} \end{cases}`} display />}
          description="最优策略是确定性的：总是选择动作值最大的动作。"
        />
        <FormulaCard
          title="值迭代形式"
          formula={<KaTeX math={String.raw`v_{k+1}(s) = \max_a q_k(s,a)`} display />}
          description="反复对 BOE 右端做贪心策略更新与值备份，可收敛到最优值函数。"
        />
      </section>

      <InteractiveDemo title="最优策略随参数变化">
        <div className="grid lg:grid-cols-[1fr_340px] gap-6">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld
              config={config}
              policy={optimalPolicy}
              values={optimalValues}
              showValues
              className="max-w-full"
            />
            <p className="mt-4 text-sm text-gray-500 text-center">
              深蓝色箭头为当前最优策略；颜色表示最优状态值
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">迭代收敛信息</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-gray-700">
                <div>值迭代步数：<span className="font-mono font-semibold">{iterations}</span></div>
                <div>目标奖励：<span className="font-mono">{config.targetReward}</span></div>
                <div>禁区奖励：<span className="font-mono">{config.forbiddenReward}</span></div>
                <div>边界奖励：<span className="font-mono">{config.boundaryReward}</span></div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">折扣因子 <KaTeX math={String.raw`\gamma`} /></CardTitle>
              </CardHeader>
              <CardContent>
                <Slider
                  value={[config.gamma]}
                  min={0}
                  max={0.99}
                  step={0.01}
                  onValueChange={([v]) => setConfig({ ...config, gamma: v })}
                />
                <div className="mt-2 text-center font-mono text-sm text-gray-700">
                  γ = {config.gamma.toFixed(2)}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">奖励设置</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="text-sm text-gray-600 mb-1">目标奖励</div>
                  <Slider
                    value={[config.targetReward]}
                    min={0.5}
                    max={3}
                    step={0.5}
                    onValueChange={([v]) => setConfig({ ...config, targetReward: v })}
                  />
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">禁区奖励</div>
                  <Slider
                    value={[config.forbiddenReward]}
                    min={-3}
                    max={-0.5}
                    step={0.5}
                    onValueChange={([v]) => setConfig({ ...config, forbiddenReward: v })}
                  />
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">边界奖励</div>
                  <Slider
                    value={[config.boundaryReward]}
                    min={-3}
                    max={-0.5}
                    step={0.5}
                    onValueChange={([v]) => setConfig({ ...config, boundaryReward: v })}
                  />
                </div>
              </CardContent>
            </Card>

            <Button onClick={reset} variant="outline" className="w-full">
              <RotateCcw className="w-4 h-4 mr-1" />
              重置参数
            </Button>
          </div>
        </div>
      </InteractiveDemo>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">观察与思考</h2>
        <ul className="space-y-3 text-gray-700">
          <li>
            <strong>折扣因子 γ：</strong>γ 越大，智能体越看重长远回报，可能更愿意绕路避开禁区；γ 越小，智能体越短视。
          </li>
          <li>
            <strong>奖励大小：</strong>当禁区惩罚足够大时，智能体会选择更保守的路径；当目标奖励较小时，可能宁愿停留不动。
          </li>
          <li>
            <strong>仿射不变性：</strong>对所有奖励同时加上一个常数，最优策略通常不变（但值函数会平移）。
          </li>
        </ul>
      </section>
    </div>
  );
}
