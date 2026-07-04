import { useState, useMemo } from 'react';
import { BarChart3, TrendingUp, ShieldAlert, Play, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import LineChart from '@/components/LineChart';
import {
  DEFAULT_CONFIG,
  type Action,
  type Policy,
  deterministicPolicy,
  solveStateValues,
  estimateStateValueMC,
} from '@/lib/rl/gridworld';

export default function Chapter02StateValuesPage() {
  const config = DEFAULT_CONFIG;
  const [policyA, setPolicyA] = useState<Policy>(() =>
    deterministicPolicy([1, 1, 3, 1, 1, 1, 1, 1, 4] as Action[])
  );
  const [policyB, setPolicyB] = useState<Policy>(() =>
    deterministicPolicy([2, 1, 2, 2, 1, 2, 2, 1, 4] as Action[])
  );
  const [mcPolicy, setMcPolicy] = useState<'A' | 'B'>('A');
  const [mcStart, setMcStart] = useState(config.startState);
  const [mcEpisodes, setMcEpisodes] = useState(100);
  const [mcData, setMcData] = useState<{ episode: number; estimate: number; trueValue: number }[]>([]);

  const valuesA = useMemo(() => solveStateValues(policyA, config), [policyA, config]);
  const valuesB = useMemo(() => solveStateValues(policyB, config), [policyB, config]);

  const selectedPolicy = mcPolicy === 'A' ? policyA : policyB;
  const selectedValues = mcPolicy === 'A' ? valuesA : valuesB;
  const trueStartValue = selectedValues[mcStart];

  function runMC() {
    const { estimates } = estimateStateValueMC(mcStart, selectedPolicy, config, mcEpisodes, 30);
    setMcData(
      estimates.map((estimate, i) => ({
        episode: i + 1,
        estimate,
        trueValue: trueStartValue,
      }))
    );
  }

  function updatePolicy(
    setter: React.Dispatch<React.SetStateAction<Policy>>,
    state: number,
    action: number
  ) {
    setter((prev) =>
      prev.map((dist, s) => {
        if (s !== state) return dist;
        const d = new Array(5).fill(0);
        d[action] = 1;
        return d;
      })
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      {/* Hero */}
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <BarChart3 className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          2.1 状态值函数
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          状态值 v(s) 是从某状态出发遵循策略的期望回报。它是最核心的策略评估指标。
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
            <KaTeX math={String.raw`v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]`} />
            依赖于策略 π 与初始状态 s。
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">回报 vs 状态值</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            确定环境下回报等于状态值；随机环境下状态值是多次回报的平均。
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">策略评估</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600">
            求解状态值的过程称为策略评估（policy evaluation）。
          </CardContent>
        </Card>
      </section>

      {/* Interactive demo */}
      <InteractiveDemo title="策略评估实验室">
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Policy A */}
          <div className="space-y-4">
            <h3 className="font-semibold text-gray-900">策略 A</h3>
            <div className="bg-gray-50 rounded-xl p-4 border border-gray-200 flex justify-center">
              <GridWorld
                config={config}
                policy={policyA}
                values={valuesA}
                showValues
                onActionClick={(s, a) => updatePolicy(setPolicyA, s, a)}
                editable
                className="max-w-[240px]"
              />
            </div>
            <div className="text-sm text-gray-600 bg-blue-50 p-3 rounded-lg border border-blue-100">
              平均状态值：{(valuesA.reduce((a, b) => a + b, 0) / valuesA.length).toFixed(3)}
            </div>
          </div>

          {/* Policy B */}
          <div className="space-y-4">
            <h3 className="font-semibold text-gray-900">策略 B</h3>
            <div className="bg-gray-50 rounded-xl p-4 border border-gray-200 flex justify-center">
              <GridWorld
                config={config}
                policy={policyB}
                values={valuesB}
                showValues
                onActionClick={(s, a) => updatePolicy(setPolicyB, s, a)}
                editable
                className="max-w-[240px]"
              />
            </div>
            <div className="text-sm text-gray-600 bg-blue-50 p-3 rounded-lg border border-blue-100">
              平均状态值：{(valuesB.reduce((a, b) => a + b, 0) / valuesB.length).toFixed(3)}
            </div>
          </div>
        </div>

        <div className="mt-6 grid md:grid-cols-2 gap-4 text-sm text-gray-700">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <strong>颜色含义：</strong>绿色越深表示状态值越大（越接近目标），红色越深表示状态值越小。
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <strong>比较：</strong>若策略 A 在所有状态上的值都不小于策略 B，则 A 更优。
          </div>
        </div>
      </InteractiveDemo>

      {/* MC convergence */}
      <InteractiveDemo title="蒙特卡洛估计：回报的平均收敛到状态值">
        <div className="grid lg:grid-cols-[1fr_320px] gap-6">
          <div>
            {mcData.length > 0 ? (
              <LineChart
                data={mcData}
                xKey="episode"
                xLabel="回合数"
                yLabel="估计值"
                series={[
                  { key: 'estimate', name: '样本平均', color: '#2563eb' },
                  { key: 'trueValue', name: '真实状态值', color: '#ef4444', strokeDasharray: '5 5' },
                ]}
                height={280}
              />
            ) : (
              <div className="h-[280px] flex items-center justify-center bg-gray-50 rounded-xl border border-gray-200 text-gray-500">
                点击「运行蒙特卡洛」查看收敛曲线
              </div>
            )}
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  设置
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="text-sm text-gray-700 mb-2">选择策略</div>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant={mcPolicy === 'A' ? 'default' : 'outline'}
                      onClick={() => setMcPolicy('A')}
                    >
                      策略 A
                    </Button>
                    <Button
                      size="sm"
                      variant={mcPolicy === 'B' ? 'default' : 'outline'}
                      onClick={() => setMcPolicy('B')}
                    >
                      策略 B
                    </Button>
                  </div>
                </div>

                <div>
                  <div className="text-sm text-gray-700 mb-2">起始状态 s{mcStart + 1}</div>
                  <div className="flex flex-wrap gap-1">
                    {Array.from({ length: 9 }, (_, i) => (
                      <button
                        key={i}
                        onClick={() => setMcStart(i)}
                        className={`w-8 h-8 rounded text-xs font-medium border transition-colors ${
                          mcStart === i
                            ? 'bg-blue-600 text-white border-blue-600'
                            : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
                        }`}
                      >
                        s{i + 1}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="text-sm text-gray-700">
                  真实状态值：
                  <span className="font-mono font-semibold ml-1">{trueStartValue.toFixed(3)}</span>
                </div>

                <div>
                  <div className="flex justify-between text-sm text-gray-700 mb-1">
                    <span>回合数</span>
                    <span className="font-mono">{mcEpisodes}</span>
                  </div>
                  <Slider
                    value={[mcEpisodes]}
                    min={10}
                    max={500}
                    step={10}
                    onValueChange={([v]) => setMcEpisodes(v)}
                  />
                </div>

                <div className="flex gap-2">
                  <Button onClick={runMC} size="sm" className="bg-blue-600 hover:bg-blue-700">
                    <Play className="w-4 h-4 mr-1" />
                    运行蒙特卡洛
                  </Button>
                  <Button
                    onClick={() => setMcData([])}
                    size="sm"
                    variant="outline"
                  >
                    <RotateCcw className="w-4 h-4 mr-1" />
                    清空
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </InteractiveDemo>

      {/* Formula summary */}
      <section className="space-y-4">
        <FormulaCard
          title="状态值函数"
          formula={<KaTeX math={String.raw`v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]`} display />}
          description="从状态 s 出发遵循策略 π 的期望折扣回报。"
        />
        <FormulaCard
          title="蒙特卡洛估计"
          formula={<KaTeX math={String.raw`v_\pi(s) \approx \frac{1}{N} \sum_{i=1}^{N} G^{(i)}`} display />}
          description="通过多次采样的回报平均，估计真实状态值。"
        />
      </section>
    </div>
  );
}
