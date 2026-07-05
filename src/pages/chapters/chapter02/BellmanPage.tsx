import { useState, useMemo } from 'react';
import { Calculator, Play, RotateCcw, TrendingUp, ShieldAlert, Info, Table2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import {
  DEFAULT_CONFIG,
  type Action,
  type Policy,
  deterministicPolicy,
  solveStateValues,
  iterateStateValues,
  computeBellmanBackup,
  computeBellmanComponents,
} from '@/lib/rl/gridworld';

export default function Chapter02BellmanPage() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [policy, setPolicy] = useState<Policy>(() =>
    deterministicPolicy([1, 1, 2, 1, 1, 2, 4, 1, 4] as Action[])
  );
  const [iterationStep, setIterationStep] = useState(0);
  const [history, setHistory] = useState<number[][]>(() => {
    const numStates = config.rows * config.cols;
    return [new Array(numStates).fill(0)];
  });
  const [showExact, setShowExact] = useState(false);
  const [selectedState, setSelectedState] = useState<number>(config.startState);
  const [activeTab, setActiveTab] = useState('iterate');

  const exactValues = useMemo(() => solveStateValues(policy, config), [policy, config]);

  const displayedValues = showExact
    ? exactValues
    : history[Math.min(iterationStep, history.length - 1)];

  const backup = useMemo(
    () => computeBellmanBackup(selectedState, displayedValues, policy, config),
    [selectedState, displayedValues, policy, config]
  );

  const { rPi, pPi } = useMemo(
    () => computeBellmanComponents(policy, config),
    [policy, config]
  );

  function updatePolicy(state: number, action: number) {
    const newPolicy = deterministicPolicy(
      policy.map((dist, s) => (s === state ? (action as Action) : (dist.indexOf(1) as Action))) as Action[]
    );
    setPolicy(newPolicy);
    resetHistory(newPolicy);
  }

  function resetHistory(newPolicy?: Policy) {
    const numStates = config.rows * config.cols;
    const p = newPolicy || policy;
    const newHistory = iterateStateValues(p, config, 0, new Array(numStates).fill(0));
    setHistory(newHistory);
    setIterationStep(0);
    setShowExact(false);
  }

  function step() {
    if (iterationStep >= history.length - 1) {
      const next = iterateStateValues(policy, config, 1, history[history.length - 1]);
      const extended = [...history, next[next.length - 1]];
      setHistory(extended);
      setIterationStep(extended.length - 1);
    } else {
      setIterationStep(iterationStep + 1);
    }
  }

  function runToConvergence() {
    const converged = iterateStateValues(policy, config, 200, new Array(config.rows * config.cols).fill(0));
    setHistory(converged);
    setIterationStep(converged.length - 1);
  }

  function reset() {
    resetHistory();
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      {/* Hero */}
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Calculator className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          2.2 贝尔曼方程
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          贝尔曼方程描述了状态值函数之间的自举关系。通过迭代备份，我们可以求解任意策略的状态值。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      {/* Formula cards */}
      <section className="space-y-4">
        <FormulaCard
          title="状态值函数"
          formula={<KaTeX math={String.raw`v_\pi(s) = \mathbb{E}[G_t | S_t = s]`} display />}
          description="从状态 s 出发，遵循策略 π 所能获得的期望折扣回报。"
        />
        <FormulaCard
          title="贝尔曼方程（元素形式）"
          formula={
            <KaTeX
              math={String.raw`v_\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a) \big[ r + \gamma v_\pi(s') \big]`}
              display
            />
          }
          description="当前状态的值等于即时奖励期望加上折扣后的下一状态值期望。"
        />
        <FormulaCard
          title="贝尔曼方程（矩阵形式）"
          formula={<KaTeX math={String.raw`v_\pi = r_\pi + \gamma P_\pi v_\pi \quad \Rightarrow \quad v_\pi = (I - \gamma P_\pi)^{-1} r_\pi`} display />}
          description="对所有状态联立，可得到线性方程组并求闭式解。"
        />
        <FormulaCard
          title="迭代求解"
          formula={<KaTeX math={String.raw`v_{k+1} = r_\pi + \gamma P_\pi v_k`} display />}
          description="反复应用贝尔曼备份直至收敛，这是策略评估的核心。"
        />
      </section>

      {/* Interactive demo */}
      <InteractiveDemo title="贝尔曼方程与策略评估">
        <div className="grid lg:grid-cols-[1fr_420px] gap-6">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld
              config={config}
              policy={policy}
              values={displayedValues}
              showValues
              highlightState={selectedState}
              onCellClick={setSelectedState}
              onActionClick={updatePolicy}
              editable
              className="max-w-full"
            />
            <p className="mt-4 text-sm text-gray-500 text-center">
              颜色深浅表示状态值大小；点击格子查看该状态的贝尔曼备份；点击箭头修改策略
            </p>
          </div>

          <div className="space-y-4">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="iterate">迭代求解</TabsTrigger>
                <TabsTrigger value="backup">备份拆解</TabsTrigger>
                <TabsTrigger value="matrix">矩阵视图</TabsTrigger>
              </TabsList>

              <TabsContent value="iterate" className="space-y-4 mt-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <TrendingUp className="w-4 h-4" />
                      迭代状态
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between text-sm text-gray-600">
                      <span>当前步数</span>
                      <span className="font-mono font-semibold">{iterationStep}</span>
                    </div>
                    <Slider
                      value={[iterationStep]}
                      min={0}
                      max={history.length - 1}
                      step={1}
                      onValueChange={([v]) => {
                        setIterationStep(v);
                        setShowExact(false);
                      }}
                    />
                    <div className="flex gap-2">
                      <Button onClick={step} size="sm" className="flex-1 bg-blue-600 hover:bg-blue-700">
                        <Play className="w-4 h-4 mr-1" />
                        迭代一步
                      </Button>
                      <Button onClick={runToConvergence} size="sm" variant="outline" className="flex-1">
                        运行到收敛
                      </Button>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="exact"
                        checked={showExact}
                        onChange={(e) => setShowExact(e.target.checked)}
                        className="rounded border-gray-300"
                      />
                      <label htmlFor="exact" className="text-sm text-gray-700">
                        显示闭式解 <KaTeX math={String.raw`(I - \gamma P)^{-1} r`} />
                      </label>
                    </div>
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
                      onValueChange={([v]) => {
                        setConfig({ ...config, gamma: v });
                        resetHistory();
                      }}
                    />
                    <div className="mt-2 text-center font-mono text-sm text-gray-700">
                      γ = {config.gamma.toFixed(2)}
                    </div>
                  </CardContent>
                </Card>

                <Button onClick={reset} variant="outline" className="w-full">
                  <RotateCcw className="w-4 h-4 mr-1" />
                  重置
                </Button>
              </TabsContent>

              <TabsContent value="backup" className="space-y-4 mt-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Info className="w-4 h-4" />
                      状态 s{selectedState + 1} 的贝尔曼备份
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="bg-gray-50 p-3 rounded-lg border border-gray-200 text-sm text-gray-700">
                      <KaTeX
                        math={String.raw`v(s) = r_\pi(s) + \gamma \sum_{s'} P_\pi(s'|s) v(s')`}
                        display
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div className="bg-blue-50 p-3 rounded-lg border border-blue-100">
                        <div className="text-gray-600">即时奖励期望</div>
                        <div className="font-mono font-semibold">{backup.immediateReward.toFixed(3)}</div>
                      </div>
                      <div className="bg-blue-50 p-3 rounded-lg border border-blue-100">
                        <div className="text-gray-600">折扣未来值</div>
                        <div className="font-mono font-semibold">{backup.futureValue.toFixed(3)}</div>
                      </div>
                    </div>

                    <div className="bg-emerald-50 p-3 rounded-lg border border-emerald-100">
                      <div className="text-gray-700">
                        备份结果：
                        <span className="font-mono font-semibold ml-1">
                          {backup.backupValue.toFixed(3)}
                        </span>
                      </div>
                      <div className="text-gray-700">
                        当前显示值：
                        <span className="font-mono font-semibold ml-1">
                          {displayedValues[selectedState].toFixed(3)}
                        </span>
                      </div>
                    </div>

                    <div>
                      <div className="text-sm text-gray-700 mb-2">下一状态贡献</div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm border-collapse">
                          <thead>
                            <tr className="border-b border-gray-200 text-gray-600">
                              <th className="py-1 pr-2 text-left">s'</th>
                              <th className="py-1 pr-2 text-left">概率</th>
                              <th className="py-1 pr-2 text-left">v(s')</th>
                              <th className="py-1 text-left">γ·P·v</th>
                            </tr>
                          </thead>
                          <tbody>
                            {backup.contributions.map((c) => (
                              <tr key={c.nextState} className="border-b border-gray-100">
                                <td className="py-1 pr-2">s{c.nextState + 1}</td>
                                <td className="py-1 pr-2 font-mono">{c.prob.toFixed(2)}</td>
                                <td className="py-1 pr-2 font-mono">{c.value.toFixed(3)}</td>
                                <td className="py-1 font-mono">
                                  {(config.gamma * c.weightedValue).toFixed(3)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="matrix" className="space-y-4 mt-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Table2 className="w-4 h-4" />
                      策略下的转移矩阵 P_π
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto max-h-[240px] overflow-y-auto">
                      <table className="w-full text-xs border-collapse">
                        <thead className="sticky top-0 bg-gray-50">
                          <tr className="border-b border-gray-200">
                            <th className="py-1 pr-2 text-left"> </th>
                            {Array.from({ length: 9 }, (_, i) => (
                              <th key={i} className="py-1 pr-2 text-left">s{i + 1}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {pPi.map((row, s) => (
                            <tr key={s} className="border-b border-gray-100">
                              <td className="py-1 pr-2 font-medium">s{s + 1}</td>
                              {row.map((p, j) => (
                                <td key={j} className="py-1 pr-2 font-mono">
                                  {p > 0 ? p.toFixed(2) : '0'}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">即时奖励期望 r_π</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-3 gap-2">
                      {rPi.map((r, s) => (
                        <div
                          key={s}
                          className="bg-gray-50 rounded border border-gray-200 p-2 text-center text-sm"
                        >
                          <div className="text-gray-500">s{s + 1}</div>
                          <div className="font-mono font-semibold">{r.toFixed(3)}</div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </InteractiveDemo>

      {/* Interpretation */}
      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">如何理解这个演示？</h2>
        <ul className="space-y-3 text-gray-700">
          <li>
            <strong>颜色：</strong>绿色越深表示状态值越大（越接近目标、奖励越好），红色越深表示状态值越小。
          </li>
          <li>
            <strong>策略：</strong>点击某个格子中的箭头，可以改变该状态执行的确定性动作。
          </li>
          <li>
            <strong>备份拆解：</strong>选择一个状态，观察贝尔曼方程如何把它拆成「即时奖励」和「折扣后的下一状态值」。
          </li>
          <li>
            <strong>矩阵视图：</strong>P_π 是策略下的状态转移矩阵，r_π 是即时奖励期望向量。
          </li>
          <li>
            <strong>迭代：</strong>每一步都按照贝尔曼方程对所有状态进行一次备份，最终收敛到真实状态值。
          </li>
          <li>
            <strong>闭式解：</strong>勾选后可与矩阵求逆得到的精确解对比。
          </li>
        </ul>
      </section>
    </div>
  );
}
