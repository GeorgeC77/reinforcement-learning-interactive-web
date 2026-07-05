import { useState, useMemo } from 'react';
import { Trophy, RotateCcw, ShieldAlert, Scale } from 'lucide-react';
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
  const [alpha, setAlpha] = useState(1);
  const [beta, setBeta] = useState(0);

  const { values, policies } = useMemo(() => {
    return valueIteration(config, 200, 1e-6);
  }, [config]);

  const optimalValues = values[values.length - 1];
  const optimalPolicy = policies[policies.length - 1];
  const iterations = values.length - 1;

  // Affine transformation demo uses the base rewards from DEFAULT_CONFIG
  const affineConfig = useMemo(
    () => ({
      ...DEFAULT_CONFIG,
      gamma: config.gamma,
      targetReward: alpha * DEFAULT_CONFIG.targetReward + beta,
      forbiddenReward: alpha * DEFAULT_CONFIG.forbiddenReward + beta,
      boundaryReward: alpha * DEFAULT_CONFIG.boundaryReward + beta,
      stepReward: alpha * DEFAULT_CONFIG.stepReward + beta,
    }),
    [alpha, beta, config.gamma]
  );

  const affineResult = useMemo(() => {
    const vi = valueIteration(affineConfig, 200, 1e-6);
    return {
      values: vi.values[vi.values.length - 1],
      policy: vi.policies[vi.policies.length - 1],
    };
  }, [affineConfig]);

  const policiesMatch = useMemo(
    () =>
      optimalPolicy.every((dist, s) =>
        dist.every((p, a) => Math.abs(p - affineResult.policy[s][a]) < 1e-6)
      ),
    [optimalPolicy, affineResult.policy]
  );

  function reset() {
    setConfig(DEFAULT_CONFIG);
    setAlpha(1);
    setBeta(0);
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
                <CardTitle className="text-base">环境奖励参数</CardTitle>
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
                <p className="text-xs text-amber-700 bg-amber-50 p-2 rounded border border-amber-100">
                  单独改变某一类奖励可能会改变最优策略。真正的仿射不变性需要所有奖励同时做同一变换，见下方演示。
                </p>
              </CardContent>
            </Card>

            <Button onClick={reset} variant="outline" className="w-full">
              <RotateCcw className="w-4 h-4 mr-1" />
              重置参数
            </Button>
          </div>
        </div>
      </InteractiveDemo>

      {/* Affine transformation module */}
      <InteractiveDemo title="奖励的仿射不变性">
        <div className="grid lg:grid-cols-[1fr_340px] gap-6">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-sm text-gray-600 mb-2">原始奖励</div>
                <GridWorld
                  config={DEFAULT_CONFIG}
                  policy={optimalPolicy}
                  values={optimalValues}
                  showValues
                  className="max-w-[180px] mx-auto"
                />
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-600 mb-2">变换后奖励</div>
                <GridWorld
                  config={affineConfig}
                  policy={affineResult.policy}
                  values={affineResult.values}
                  showValues
                  className="max-w-[180px] mx-auto"
                />
              </div>
            </div>
            <p className="mt-4 text-sm text-gray-500 text-center">
              所有奖励按 r' = α·r + β 变换后，最优策略通常保持不变
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Scale className="w-4 h-4" />
                  仿射变换参数
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm text-gray-700 mb-1">
                    <span>α（缩放，必须 &gt; 0）</span>
                    <span className="font-mono">{alpha.toFixed(1)}</span>
                  </div>
                  <Slider
                    value={[alpha]}
                    min={0.1}
                    max={3}
                    step={0.1}
                    onValueChange={([v]) => setAlpha(v)}
                  />
                </div>
                <div>
                  <div className="flex justify-between text-sm text-gray-700 mb-1">
                    <span>β（平移）</span>
                    <span className="font-mono">{beta.toFixed(1)}</span>
                  </div>
                  <Slider
                    value={[beta]}
                    min={-2}
                    max={2}
                    step={0.5}
                    onValueChange={([v]) => setBeta(v)}
                  />
                </div>

                <div className="text-sm text-gray-700 bg-gray-50 p-3 rounded-lg border border-gray-200">
                  <div>r' = {alpha.toFixed(1)}·r + {beta.toFixed(1)}</div>
                  <div className="mt-1">目标：{(alpha * DEFAULT_CONFIG.targetReward + beta).toFixed(1)}</div>
                  <div>禁区：{(alpha * DEFAULT_CONFIG.forbiddenReward + beta).toFixed(1)}</div>
                  <div>边界：{(alpha * DEFAULT_CONFIG.boundaryReward + beta).toFixed(1)}</div>
                  <div>普通步：{(alpha * DEFAULT_CONFIG.stepReward + beta).toFixed(1)}</div>
                </div>

                <div
                  className={`text-sm p-3 rounded-lg border ${
                    policiesMatch
                      ? 'bg-green-50 text-green-800 border-green-200'
                      : 'bg-red-50 text-red-800 border-red-200'
                  }`}
                >
                  {policiesMatch
                    ? '✓ 最优策略在变换前后保持一致'
                    : '✗ 最优策略发生变化（可能由于数值精度或 α ≤ 0）'}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </InteractiveDemo>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">观察与思考</h2>
        <ul className="space-y-3 text-gray-700">
          <li>
            <strong>折扣因子 γ：</strong>γ 越大，智能体越重视长期累计回报，可能愿意承担短期负奖励以换取更高长期回报；γ 越小越短视。在禁区惩罚较小时，γ 较大的策略有时反而会穿过禁区；只有禁区惩罚足够大时，最优策略才会明显绕开禁区。
          </li>
          <li>
            <strong>奖励大小：</strong>当禁区惩罚足够大时，智能体会选择更保守的路径；当目标奖励较小时，可能宁愿停留不动。
          </li>
          <li>
            <strong>仿射不变性：</strong>对所有奖励同时做同一变换 r' = α·r + β（α &gt; 0），最优策略不变，但最优值函数会相应缩放与平移。单独改变某一类奖励不属于仿射不变性。
          </li>
        </ul>
      </section>
    </div>
  );
}
