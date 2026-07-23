import { useState, useMemo } from 'react';
import { Clock, ShieldAlert, Play } from 'lucide-react';
import { Button } from '@/components/ui/button';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import ConceptAccordion from '@/components/ConceptAccordion';
import TdDemo from './TdDemo';
import {
  EPISODIC_PATH_CONFIG,
  ACTION_NAMES,
  deterministicPolicy,
  greedyPolicy,
  epsilonGreedyPolicy,
  actionValueToStateValue,
  tdZeroPrediction,
  sarsa,
  qLearning,
  nStepSarsa,
  type Action,
  type TDUpdateRecord,
} from '@/lib/rl/gridworld';
import type { AlgorithmKind } from './TdDemo';

type UnifiedAlgo = Exclude<AlgorithmKind, 'expected' | 'sarsa-lambda' | 'td-lambda'>;

const GOAL_POLICY: Action[] = [1, 2, 4, 1, 2, 4, 4, 1, 4];

const UNIFIED_ROWS: {
  key: UnifiedAlgo;
  label: string;
  target: string;
  equation: string;
  onPolicy: string;
  bootstrap: string;
  steps: string;
  mcRelation: string;
  dpRelation: string;
}[] = [
  {
    key: 'td0',
    label: 'TD(0)',
    target: String.raw`r_{t+1} + \gamma v(s_{t+1})`,
    equation: String.raw`v = r_\pi + \gamma P_\pi v`,
    onPolicy: 'on-policy（固定策略）',
    bootstrap: '是',
    steps: '1',
    mcRelation: 'n=1 的 n-step',
    dpRelation: '用采样转移代替 P_π',
  },
  {
    key: 'sarsa',
    label: 'Sarsa',
    target: String.raw`r_{t+1} + \gamma q(s_{t+1}, a_{t+1})`,
    equation: String.raw`q_\pi = r + \gamma P_\pi q_\pi`,
    onPolicy: 'on-policy',
    bootstrap: '是',
    steps: '1',
    mcRelation: '用实际执行动作采样',
    dpRelation: '同策略 Bellman 方程',
  },
  {
    key: 'nstep',
    label: 'n-step Sarsa',
    target: String.raw`G_t^{(n)} = \sum_{i=1}^{n}\gamma^{i-1}r_{t+i} + \gamma^n q(s_{t+n}, a_{t+n})`,
    equation: String.raw`q_\pi = \mathbb{E}_\pi[G_t^{(n)}]`,
    onPolicy: 'on-policy',
    bootstrap: '是',
    steps: 'n',
    mcRelation:
      '在有限 episodic 任务中，当 n 覆盖到回合终点的全部剩余步数时，n-step return 等于 Monte Carlo return；continuing 或人工截断任务下不能无条件写成 n=∞ 即 MC。',
    dpRelation: 'n=1 时为 Sarsa',
  },
  {
    key: 'qlearning',
    label: 'Q-learning',
    target: String.raw`r_{t+1} + \gamma \max_a q(s_{t+1}, a)`,
    equation: String.raw`q^* = r + \gamma P \max_a q^*`,
    onPolicy: 'off-policy',
    bootstrap: '是',
    steps: '1',
    mcRelation: 'target 不依赖实际动作',
    dpRelation: 'Bellman optimality equation',
  },
];

function firstUpdate(algo: UnifiedAlgo): TDUpdateRecord | null {
  const config = EPISODIC_PATH_CONFIG;
  const policy = deterministicPolicy(GOAL_POLICY, 5);
  let result;
  switch (algo) {
    case 'td0':
      result = tdZeroPrediction(policy, config, 0.2, 20, 1, 1);
      break;
    case 'sarsa':
      result = sarsa(config, 0.2, 0.3, 'fixed', 20, 1, 1);
      break;
    case 'nstep':
      result = nStepSarsa(config, 0.2, 0.3, 'fixed', 3, 20, 1, 1);
      break;
    case 'qlearning':
      result = qLearning(config, 0.2, 0.3, 'fixed', 20, 1, 1);
      break;
    default:
      return null;
  }
  return result.updates[0] ?? null;
}

export default function Chapter07TdPage() {
  const [demoAlgo, setDemoAlgo] = useState<UnifiedAlgo>('td0');
  const demoUpdate = useMemo(() => firstUpdate(demoAlgo), [demoAlgo]);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Clock className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">第 7 章 时序差分方法</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          TD(0)、Sarsa、n-step Sarsa、Q-learning：在采样与自举之间架起桥梁。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="TD(0) 预测"
          formula={<KaTeX math={String.raw`v(s_t) \leftarrow v(s_t) + \alpha \bigl[ r_{t+1} + \gamma v(s_{t+1}) - v(s_t) \bigr]`} display />}
          description="每接收一个转移样本就更新一次状态值，是最简单的自举方法。"
        />
        <FormulaCard
          title="Sarsa（同策略控制）"
          formula={<KaTeX math={String.raw`q(s_t,a_t) \leftarrow q(s_t,a_t) + \alpha \bigl[ r_{t+1} + \gamma q(s_{t+1},a_{t+1}) - q(s_t,a_t) \bigr]`} display />}
          description="TD 目标中的下一个动作 a_{t+1} 来自当前策略实际采样。"
        />
        <FormulaCard
          title="n-step Sarsa"
          formula={
            <KaTeX
              math={String.raw`G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n q(s_{t+n}, a_{t+n})`}
              display
            />
          }
          description="在 TD（n=1）和 MC（n=∞）之间做 bias-variance 权衡。"
        />
        <FormulaCard
          title="Q-learning（异策略控制）"
          formula={
            <KaTeX
              math={String.raw`q(s_t,a_t) \leftarrow q(s_t,a_t) + \alpha \bigl[ r_{t+1} + \gamma \max_a q(s_{t+1},a) - q(s_t,a_t) \bigr]`}
              display
            />
          }
          description="TD 目标使用下一个状态的最大动作值，即使实际执行的动作不是它。"
        />
      </section>

      <TdDemo
        title="TD 算法动态教学"
        subtitle="逐 transition 观察更新，切换行为策略价值与贪心派生价值，比较 prediction 与控制面板。"
        algorithms={[
          { key: 'td0', label: 'TD(0)', category: 'main' },
          { key: 'sarsa', label: 'Sarsa', category: 'main' },
          { key: 'nstep', label: 'n-step', category: 'main' },
          { key: 'qlearning', label: 'Q-learning', category: 'main' },
        ]}
        defaultAlgorithm="td0"
        persistKey="ch07.td"
      />

      <InteractiveDemo title="TD 统一视角">
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="bg-gray-100 text-gray-700">
                <th className="border p-2 text-left">算法</th>
                <th className="border p-2 text-left">sample target</th>
                <th className="border p-2 text-left">对应方程</th>
                <th className="border p-2 text-left">on/off-policy</th>
                <th className="border p-2 text-left">bootstrap</th>
                <th className="border p-2 text-left">步数</th>
                <th className="border p-2 text-left">与 MC 关系</th>
                <th className="border p-2 text-left">与 DP 关系</th>
              </tr>
            </thead>
            <tbody>
              {UNIFIED_ROWS.map((row) => (
                <tr
                  key={row.key}
                  className={`cursor-pointer hover:bg-blue-50 ${demoAlgo === row.key ? 'bg-blue-100' : ''}`}
                  onClick={() => setDemoAlgo(row.key)}
                >
                  <td className="border p-2 font-medium">{row.label}</td>
                  <td className="border p-2">
                    <KaTeX math={row.target} />
                  </td>
                  <td className="border p-2">
                    <KaTeX math={row.equation} />
                  </td>
                  <td className="border p-2">{row.onPolicy}</td>
                  <td className="border p-2">{row.bootstrap}</td>
                  <td className="border p-2">{row.steps}</td>
                  <td className="border p-2">{row.mcRelation}</td>
                  <td className="border p-2">{row.dpRelation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-6 grid lg:grid-cols-[1fr_360px] gap-6">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            {demoUpdate ? (
              <>
                <GridWorld
                  config={EPISODIC_PATH_CONFIG}
                  policy={
                    demoAlgo === 'td0'
                      ? deterministicPolicy(GOAL_POLICY, 5)
                      : demoAlgo === 'qlearning'
                        ? greedyPolicy(demoUpdate.qAfter!)
                        : epsilonGreedyPolicy(demoUpdate.qAfter!, 0.3)
                  }
                  values={
                    demoAlgo === 'td0'
                      ? demoUpdate.valuesAfter!
                      : actionValueToStateValue(demoUpdate.qAfter!)
                  }
                  showValues
                  highlightState={demoUpdate.state}
                  highlightNextState={demoUpdate.nextState}
                  highlightUpdatedState={demoUpdate.state}
                  highlightAction={demoAlgo === 'td0' ? null : { state: demoUpdate.state, action: demoUpdate.action }}
                  className="max-w-full"
                />
                <p className="mt-4 text-sm text-gray-500 text-center">
                  {demoAlgo} 第一次更新：s{demoUpdate.state + 1} · {ACTION_NAMES[demoUpdate.action]} → s{demoUpdate.nextState + 1}
                </p>
              </>
            ) : (
              <p className="text-gray-500">点击表格行播放一次实际更新</p>
            )}
          </div>

          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {UNIFIED_ROWS.map((row) => (
                <Button
                  key={row.key}
                  size="sm"
                  variant={demoAlgo === row.key ? 'default' : 'outline'}
                  onClick={() => setDemoAlgo(row.key)}
                >
                  <Play className="w-3 h-3 mr-1" />
                  {row.label}
                </Button>
              ))}
            </div>
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <div className="text-sm text-gray-600 mb-2">高亮 target</div>
              <KaTeX math={UNIFIED_ROWS.find((r) => r.key === demoAlgo)?.target ?? ''} display />
            </div>
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <div className="text-sm text-gray-600 mb-2">对应方程 / BOE</div>
              <KaTeX math={UNIFIED_ROWS.find((r) => r.key === demoAlgo)?.equation ?? ''} display />
            </div>
          </div>
        </div>
      </InteractiveDemo>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">小结与 Q&A</h2>
        <ConceptAccordion
          items={[
            {
              id: 'summary',
              title: '本章小结',
              content: (
                <ul className="list-disc pl-5 space-y-2">
                  <li>TD 算法是随机逼近求解贝尔曼/贝尔曼最优方程的特例。</li>
                  <li>TD(0) 每步更新，是在线学习的典型代表。</li>
                  <li>Sarsa 同策略、Q-learning 异策略，更新机制不同。</li>
                  <li>n-step 方法在 TD 和 MC 之间做 bias-variance 权衡。</li>
                  <li>Expected Sarsa 与 TD(λ)、Sarsa(λ) 见“补充与拓展”页面。</li>
                </ul>
              ),
            },
            {
              id: 'qa1',
              title: 'Q: TD 与 MC 的核心区别是什么？',
              content:
                'MC 需要等一个完整回合后才能更新；TD 每走一步就可以用自举目标更新，因此可以在线学习。',
            },
            {
              id: 'qa2',
              title: 'Q: 同策略与异策略控制有什么区别？',
              content:
                'Sarsa 用实际执行的下一个动作构造 TD 目标；Q-learning 用下一个状态的最大 Q 值构造目标，因此可以离策略数据学习。',
            },
          ]}
        />
      </section>
    </div>
  );
}
