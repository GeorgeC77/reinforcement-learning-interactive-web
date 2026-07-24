import { useMemo, useState } from 'react';
import { Compass, ShieldAlert } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import LineChart from '@/components/LineChart';
import ConceptAccordion from '@/components/ConceptAccordion';
import SeedControl from '@/components/SeedControl';
import { usePersistentState } from '@/hooks/usePersistentState';
import { runBandit, type BanditArm, type BanditStrategy } from '@/lib/rl/bandits';
import { dynaQ } from '@/lib/rl/dyna';
import {
  EPISODIC_PATH_CONFIG,
  greedyPolicy,
  actionValueToStateValue,
  ACTION_NAMES,
} from '@/lib/rl/gridworld';

const DEFAULT_ARMS: BanditArm[] = [
  { mean: 0.2, std: 1 },
  { mean: 0.5, std: 1 },
  { mean: 0.9, std: 1 },
  { mean: 0.4, std: 1 },
  { mean: 0.1, std: 1 },
];

const STRATEGY_LABELS: Record<BanditStrategy, string> = {
  'eps-greedy': 'ε-贪心',
  ucb1: 'UCB1',
  softmax: 'Boltzmann softmax',
};

export default function Chapter11ExplorationPage() {
  const [activeTab, setActiveTab] = useState('bandit');

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Compass className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">11.1 探索与规划</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          探索-利用权衡决定智能体多快发现最优动作；模型学习则让每一点真实经验发挥多次作用。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="遗憾（regret）"
          formula={<KaTeX math={String.raw`R_T = T\mu^* - \sum_{t=1}^{T} \mu_{a_t}, \quad \mu^* = \max_a \mu_a`} display />}
          description="累计遗憾衡量探索策略离“总是选最优臂”差多远，是探索策略的核心评价指标。"
        />
        <FormulaCard
          title="UCB1 动作选择"
          formula={<KaTeX math={String.raw`a_t = \arg\max_a \left[ Q(a) + c\sqrt{\frac{\ln t}{N(a)}} \right]`} display />}
          description="对被探索少的臂给出乐观加成：不确定性越大，越值得再试。"
        />
        <FormulaCard
          title="Dyna 架构"
          formula={<KaTeX math={String.raw`\text{真实经验} \rightarrow \text{直接 RL 更新} + \text{模型学习} \rightarrow \text{规划更新}`} display />}
          description="模型把每条真实转移扩展成多条模拟转移，价值函数收敛所需的真实样本大幅减少。"
        />
      </section>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="bandit">多臂老虎机：探索策略对比</TabsTrigger>
          <TabsTrigger value="dyna">Dyna-Q：模型与规划</TabsTrigger>
        </TabsList>
        <TabsContent value="bandit" className="mt-4">
          <BanditDemo />
        </TabsContent>
        <TabsContent value="dyna" className="mt-4">
          <DynaDemo />
        </TabsContent>
      </Tabs>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">小结与 Q&A</h2>
        <ConceptAccordion
          items={[
            {
              id: 'summary',
              title: '本章小结',
              content: (
                <ul className="list-disc pl-5 space-y-2">
                  <li>ε-贪心以固定概率随机探索，简单但长期仍浪费遗憾在差臂上。</li>
                  <li>UCB 按“置信上界”乐观探索，对不确定的臂优先尝试，遗憾增长更慢。</li>
                  <li>softmax 按价值比例探索，温度 τ 控制探索强度。</li>
                  <li>Dyna 把模型学习（估计 p、r）与规划（模拟转移更新）结合，同样的真实经验获得多倍更新。</li>
                </ul>
              ),
            },
            {
              id: 'qa-ucb',
              title: 'Q: UCB 的 ln t / N(a) 项起什么作用？',
              content:
                '它是置信上界的宽度：臂被拉得越少（N 小）、时间越久（ln t 大），上界越宽，该臂越可能被重新尝试。随着 N 增大，上界收紧，策略逐渐转为利用。',
            },
            {
              id: 'qa-dyna',
              title: 'Q: 规划步数越多越好吗？',
              content:
                '不完全是。模型有误差时，过多规划会把误差“巩固”进价值函数；且规划有计算成本。中等规划步数（如每步 5–10 次）通常性价比最高。',
            },
          ]}
        />
      </section>
    </div>
  );
}

// ------------------- Bandit demo -------------------
function BanditDemo() {
  const [steps, setSteps] = usePersistentState('ch11.bandit.steps', 500);
  const [epsilon, setEpsilon] = usePersistentState('ch11.bandit.epsilon', 0.1);
  const [tau, setTau] = usePersistentState('ch11.bandit.tau', 0.3);
  const [ucbC, setUcbC] = usePersistentState('ch11.bandit.ucbC', 1.5);
  const [seed, setSeed] = usePersistentState('ch11.bandit.seed', 1);
  const [detail, setDetail] = useState<BanditStrategy>('ucb1');

  // Run all three strategies with the same seed for a fair comparison.
  const results = useMemo(() => {
    const strategies: BanditStrategy[] = ['eps-greedy', 'ucb1', 'softmax'];
    return strategies.map((strategy) =>
      runBandit(DEFAULT_ARMS, { strategy, steps, epsilon, tau, ucbC, seed })
    );
  }, [steps, epsilon, tau, ucbC, seed]);

  const rateChart = useMemo(() => {
    const n = Math.min(steps, 400);
    const rows: Record<string, number>[] = [];
    for (let i = 0; i < n; i++) {
      const row: Record<string, number> = { t: results[0].steps[i].t };
      results.forEach((r, j) => {
        row[['eps', 'ucb', 'soft'][j]] = r.steps[i].optimalRate;
      });
      rows.push(row);
    }
    return rows;
  }, [results, steps]);

  const regretChart = useMemo(() => {
    const n = Math.min(steps, 400);
    const rows: Record<string, number>[] = [];
    for (let i = 0; i < n; i++) {
      const row: Record<string, number> = { t: results[0].steps[i].t };
      results.forEach((r, j) => {
        row[['eps', 'ucb', 'soft'][j]] = r.steps[i].regret;
      });
      rows.push(row);
    }
    return rows;
  }, [results, steps]);

  const detailIdx = (['eps-greedy', 'ucb1', 'softmax'] as BanditStrategy[]).indexOf(detail);
  const detailResult = results[detailIdx];

  return (
    <InteractiveDemo title="多臂老虎机：三种探索策略同台对比">
      <div className="grid lg:grid-cols-[1fr_320px] gap-6">
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">最优动作选择率（同种子，越靠上越好）</CardTitle>
            </CardHeader>
            <CardContent>
              <LineChart
                data={rateChart}
                xKey="t"
                xLabel="步数 t"
                yLabel="最优动作率"
                series={[
                  { key: 'eps', name: 'ε-贪心', color: '#2563eb' },
                  { key: 'ucb', name: 'UCB1', color: '#ef4444' },
                  { key: 'soft', name: 'softmax', color: '#22c55e' },
                ]}
                height={200}
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">累计遗憾（越低越好）</CardTitle>
            </CardHeader>
            <CardContent>
              <LineChart
                data={regretChart}
                xKey="t"
                xLabel="步数 t"
                yLabel="累计遗憾"
                series={[
                  { key: 'eps', name: 'ε-贪心', color: '#2563eb' },
                  { key: 'ucb', name: 'UCB1', color: '#ef4444' },
                  { key: 'soft', name: 'softmax', color: '#22c55e' },
                ]}
                height={200}
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">策略细节：{STRATEGY_LABELS[detail]}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                {DEFAULT_ARMS.map((arm, a) => (
                  <div
                    key={a}
                    className={`rounded p-2 text-center border ${
                      a === detailResult.optimalAction ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'
                    }`}
                  >
                    <div className="text-xs text-gray-500">臂 {a + 1}</div>
                    <div className="font-mono text-xs">μ={arm.mean.toFixed(1)}</div>
                    <div className="font-mono font-semibold">Q={detailResult.finalEstimates[a].toFixed(2)}</div>
                    <div className="text-xs text-gray-500">N={detailResult.actionCounts[a]}</div>
                  </div>
                ))}
              </div>
              <div className="text-xs text-gray-500">
                绿色为最优臂；Q 为学习到的估计，N 为被拉次数。UCB 会给 N 小的臂更大的探索加成。
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1"><span>步数</span><span className="font-mono">{steps}</span></div>
                <Slider value={[steps]} min={100} max={2000} step={100} onValueChange={([v]) => setSteps(v)} />
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1"><span>ε（ε-贪心）</span><span className="font-mono">{epsilon.toFixed(2)}</span></div>
                <Slider value={[epsilon]} min={0} max={0.5} step={0.01} onValueChange={([v]) => setEpsilon(v)} />
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1"><span>τ（softmax）</span><span className="font-mono">{tau.toFixed(2)}</span></div>
                <Slider value={[tau]} min={0.05} max={2} step={0.05} onValueChange={([v]) => setTau(v)} />
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1"><span>c（UCB）</span><span className="font-mono">{ucbC.toFixed(2)}</span></div>
                <Slider value={[ucbC]} min={0.2} max={4} step={0.1} onValueChange={([v]) => setUcbC(v)} />
              </div>
              <SeedControl seed={seed} onChange={setSeed} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">细节面板策略</CardTitle>
            </CardHeader>
            <CardContent>
              <Select value={detail} onValueChange={(v) => setDetail(v as BanditStrategy)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="eps-greedy">ε-贪心</SelectItem>
                  <SelectItem value="ucb1">UCB1</SelectItem>
                  <SelectItem value="softmax">Boltzmann softmax</SelectItem>
                </SelectContent>
              </Select>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- Dyna-Q demo -------------------
const PLANNING_OPTIONS = [0, 1, 5, 10, 25];

function DynaDemo() {
  const config = EPISODIC_PATH_CONFIG;
  const [planningSteps, setPlanningSteps] = usePersistentState('ch11.dyna.planningSteps', 10);
  const [alpha, setAlpha] = usePersistentState('ch11.dyna.alpha', 0.2);
  const [epsilon, setEpsilon] = usePersistentState('ch11.dyna.epsilon', 0.1);
  const [episodes, setEpisodes] = usePersistentState('ch11.dyna.episodes', 60);
  const [seed, setSeed] = usePersistentState('ch11.dyna.seed', 1);
  const [selectedState, setSelectedState] = useState(0);

  const result = useMemo(
    () => dynaQ(config, { episodes, maxSteps: 50, alpha, epsilon, planningSteps, seed }),
    [config, episodes, alpha, epsilon, planningSteps, seed]
  );
  const baseline = useMemo(
    () => dynaQ(config, { episodes, maxSteps: 50, alpha, epsilon, planningSteps: 0, seed }),
    [config, episodes, alpha, epsilon, seed]
  );

  const rmseChart = useMemo(() => {
    return result.episodes.map((e, i) => ({
      episode: e.episode,
      dyna: e.qRmse,
      plain: baseline.episodes[i].qRmse,
      accuracy: e.modelAccuracy,
    }));
  }, [result, baseline]);

  const lengthChart = useMemo(() => {
    return result.episodes.map((e, i) => ({
      episode: e.episode,
      dyna: e.steps,
      plain: baseline.episodes[i].steps,
    }));
  }, [result, baseline]);

  const finalPolicy = useMemo(() => greedyPolicy(result.finalQ), [result.finalQ]);
  const finalValues = useMemo(() => actionValueToStateValue(result.finalQ), [result.finalQ]);

  return (
    <InteractiveDemo title="Dyna-Q：真实经验 + 模型规划">
      <div className="grid lg:grid-cols-[1fr_320px] gap-6">
        <div className="space-y-4">
          <div className="flex flex-col items-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld config={config} policy={finalPolicy} values={finalValues} showValues className="max-w-full" />
            <p className="mt-3 text-sm text-gray-500 text-center">
              训练结束后的贪心策略与状态值（planning = {planningSteps}）
            </p>
          </div>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">q* RMSE 与模型准确率（规划 vs 无规划）</CardTitle>
            </CardHeader>
            <CardContent>
              <LineChart
                data={rmseChart}
                xKey="episode"
                xLabel="回合"
                yLabel="RMSE / 准确率"
                series={[
                  { key: 'dyna', name: `Dyna (n=${planningSteps})`, color: '#2563eb' },
                  { key: 'plain', name: 'Q-learning (n=0)', color: '#ef4444', strokeDasharray: '6 3' },
                  { key: 'accuracy', name: '模型准确率', color: '#22c55e' },
                ]}
                height={200}
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">每回合步数（越低越快到达目标）</CardTitle>
            </CardHeader>
            <CardContent>
              <LineChart
                data={lengthChart}
                xKey="episode"
                xLabel="回合"
                yLabel="回合长度"
                series={[
                  { key: 'dyna', name: `Dyna (n=${planningSteps})`, color: '#2563eb' },
                  { key: 'plain', name: 'Q-learning (n=0)', color: '#ef4444', strokeDasharray: '6 3' },
                ]}
                height={180}
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">学到的模型：状态 s{selectedState + 1}</CardTitle>
            </CardHeader>
            <CardContent className="text-sm">
              <div className="flex gap-2 mb-3 flex-wrap">
                {Array.from({ length: 8 }, (_, i) => (
                  <button
                    key={i}
                    onClick={() => setSelectedState(i)}
                    className={`w-8 h-8 rounded text-xs border ${
                      selectedState === i ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-200'
                    }`}
                  >
                    s{i + 1}
                  </button>
                ))}
              </div>
              <table className="w-full text-xs text-left">
                <thead className="text-gray-500 border-b">
                  <tr><th className="py-1">动作</th><th>模型 s′</th><th>模型 r̄</th><th>访问次数</th></tr>
                </thead>
                <tbody className="text-gray-700">
                  {ACTION_NAMES.map((name, a) => (
                    <tr key={a} className="border-b last:border-0">
                      <td className="py-1">{name}</td>
                      <td>{result.modelNextState[selectedState][a] >= 0 ? `s${result.modelNextState[selectedState][a] + 1}` : '—'}</td>
                      <td className="font-mono">{result.visitCounts[selectedState][a] > 0 ? result.modelReward[selectedState][a].toFixed(2) : '—'}</td>
                      <td className="font-mono">{result.visitCounts[selectedState][a]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm text-gray-700 block mb-1">规划步数 n（每真实步）</label>
                <Select value={String(planningSteps)} onValueChange={(v) => setPlanningSteps(Number(v))}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {PLANNING_OPTIONS.map((n) => (
                      <SelectItem key={n} value={String(n)}>n = {n}{n === 0 ? '（纯 Q-learning）' : ''}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1"><span>学习率 α</span><span className="font-mono">{alpha.toFixed(2)}</span></div>
                <Slider value={[alpha]} min={0.05} max={1} step={0.05} onValueChange={([v]) => setAlpha(v)} />
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1"><span>探索率 ε</span><span className="font-mono">{epsilon.toFixed(2)}</span></div>
                <Slider value={[epsilon]} min={0} max={0.5} step={0.05} onValueChange={([v]) => setEpsilon(v)} />
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-1"><span>回合数</span><span className="font-mono">{episodes}</span></div>
                <Slider value={[episodes]} min={20} max={200} step={10} onValueChange={([v]) => setEpisodes(v)} />
              </div>
              <SeedControl seed={seed} onChange={setSeed} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">统计</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-1">
              <div>真实环境步数：<span className="font-mono font-semibold">{result.totalEnvSteps}</span></div>
              <div>规划更新步数：<span className="font-mono font-semibold">{result.totalPlanningSteps}</span></div>
              <div>最终模型准确率：<span className="font-mono font-semibold">{(result.episodes[result.episodes.length - 1].modelAccuracy * 100).toFixed(1)}%</span></div>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}
