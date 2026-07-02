import { useState, useCallback, useMemo } from 'react';
import { Dices, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import {
  DEFAULT_CONFIG,
  type Action,
  generateTrajectory,
  discountedReturn,
  epsilonGreedyPolicy,
  actionValueToStateValue,
} from '@/lib/rl/gridworld';

export default function Chapter05MonteCarloPage() {
  const config = DEFAULT_CONFIG;
  const [epsilon, setEpsilon] = useState(0.3);
  const [episodeCount, setEpisodeCount] = useState(0);

  // Q-table: q[state][action]
  const [q, setQ] = useState<number[][]>(() =>
    Array.from({ length: config.rows * config.cols }, () => new Array(5).fill(0))
  );
  const [visitCounts, setVisitCounts] = useState<number[][]>(() =>
    Array.from({ length: config.rows * config.cols }, () => new Array(5).fill(0))
  );

  const policy = useMemo(() => epsilonGreedyPolicy(q, epsilon), [q, epsilon]);
  const stateValues = useMemo(() => actionValueToStateValue(q), [q]);

  const runEpisodes = useCallback(
    (n: number) => {
      let curQ = q.map((row) => [...row]);
      let curCounts = visitCounts.map((row) => [...row]);
      let curPolicy = epsilonGreedyPolicy(curQ, epsilon);

      for (let i = 0; i < n; i++) {
        const startState = Math.floor(Math.random() * config.rows * config.cols);
        const startAction = Math.floor(Math.random() * 5) as Action;
        const traj = generateTrajectory(startState, curPolicy, config, 30, startAction);

        const visited = new Set<string>();
        for (let t = 0; t < traj.length; t++) {
          const s = traj[t].state;
          const a = traj[t].action;
          const key = `${s},${a}`;
          if (visited.has(key)) continue;
          visited.add(key);

          const g = discountedReturn(traj.slice(t), config.gamma);
          curCounts[s][a] += 1;
          const alpha = 1 / curCounts[s][a];
          curQ[s][a] += alpha * (g - curQ[s][a]);
        }

        curPolicy = epsilonGreedyPolicy(curQ, epsilon);
      }

      setQ(curQ);
      setVisitCounts(curCounts);
      setEpisodeCount((c) => c + n);
    },
    [q, visitCounts, epsilon, config]
  );

  function reset() {
    setQ(
      Array.from({ length: config.rows * config.cols }, () => new Array(5).fill(0))
    );
    setVisitCounts(
      Array.from({ length: config.rows * config.cols }, () => new Array(5).fill(0))
    );
    setEpisodeCount(0);
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Dices className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          5.1 蒙特卡洛方法
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          通过采样完整回合，用样本平均更新动作值函数，再构造 ε-贪心策略。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="蒙特卡洛回报"
          formula={<KaTeX math={String.raw`G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots + \gamma^{T-t-1} r_T`} display />}
          description="从时刻 t 开始，把折扣后的未来奖励相加得到一个样本回报。"
        />
        <FormulaCard
          title="样本平均更新"
          formula={<KaTeX math={String.raw`q(s,a) \leftarrow q(s,a) + \frac{1}{N(s,a)} \bigl(G - q(s,a)\bigr)`} display />}
          description="每遇到一次 (s,a)，用新样本逐步修正估计。"
        />
        <FormulaCard
          title="ε-贪心策略"
          formula={
            <KaTeX
              math={String.raw`\pi(a|s) = \begin{cases} 1-\varepsilon + \frac{\varepsilon}{|\mathcal{A}|}, & a = \arg\max_{a'} q(s,a') \\ \frac{\varepsilon}{|\mathcal{A}|}, & \text{否则} \end{cases}`}
              display
            />
          }
          description="大部分时候选择当前最优动作，偶尔随机探索。"
        />
      </section>

      <InteractiveDemo title="MC ε-Greedy 策略改进">
        <div className="grid lg:grid-cols-[1fr_340px] gap-6">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld
              config={config}
              policy={policy}
              values={stateValues}
              showValues
              className="max-w-full"
            />
            <p className="mt-4 text-sm text-gray-500 text-center">
              箭头粗细表示动作概率；颜色表示当前状态值估计
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">统计信息</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-gray-700">
                <div>已采样回合：<span className="font-mono font-semibold">{episodeCount}</span></div>
                <div>折扣因子：<span className="font-mono">{config.gamma}</span></div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">探索参数 <KaTeX math={String.raw`\varepsilon`} /></CardTitle>
              </CardHeader>
              <CardContent>
                <Slider
                  value={[epsilon]}
                  min={0}
                  max={1}
                  step={0.05}
                  onValueChange={([v]) => setEpsilon(v)}
                />
                <div className="mt-2 text-center font-mono text-sm text-gray-700">
                  ε = {epsilon.toFixed(2)}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">采样控制</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button size="sm" onClick={() => runEpisodes(1)} className="w-full">
                  运行 1 个回合
                </Button>
                <Button size="sm" variant="outline" onClick={() => runEpisodes(10)} className="w-full">
                  运行 10 个回合
                </Button>
                <Button size="sm" variant="outline" onClick={() => runEpisodes(100)} className="w-full">
                  运行 100 个回合
                </Button>
                <Button size="sm" variant="outline" onClick={reset} className="w-full">
                  重置估计
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </InteractiveDemo>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">观察与思考</h2>
        <ul className="space-y-3 text-gray-700">
          <li>
            <strong>探索与利用：</strong>ε 越大，智能体越倾向于随机探索；ε 越小，策略越接近贪心。
          </li>
          <li>
            <strong>样本平均：</strong>回合数越多，(s,a) 的访问次数越多，q 估计越稳定。
          </li>
          <li>
            <strong>探索起点：</strong>本演示每回合从随机 (s,a) 开始，保证所有动作值都能被更新。
          </li>
        </ul>
      </section>
    </div>
  );
}
