import { useState, useMemo } from 'react';
import { Scale, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
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
import {
  DEFAULT_CONFIG,
  ACTION_NAMES,
  type Policy,
  greedyPolicy,
  estimateTrueActionValues,
  offPolicyMCEvaluation,
  qTableRMSE,
  policyWeightedStateValues,
  solveStateValues,
  type TrajectoryStep,
} from '@/lib/rl/gridworld';

type EstimateType = 'ordinary' | 'weighted';

export default function Chapter05OffPolicyMCPage() {
  const config = DEFAULT_CONFIG;
  const qStar = useMemo(() => estimateTrueActionValues(config), [config]);
  const targetPolicy = useMemo(() => greedyPolicy(qStar), [qStar]);

  const [epsilon, setEpsilon] = useState(0.5);
  const [episodesPerPair, setEpisodesPerPair] = useState(20);
  const [horizonT, setHorizonT] = useState(30);
  const [type, setType] = useState<EstimateType>('ordinary');
  const [result, setResult] = useState<{
    qValues: number[][];
    qHistory: number[][][];
    lastTrajectory: TrajectoryStep[];
    lastRho: number;
  } | null>(null);

  // Behavior policy is an epsilon-soft version of the deterministic target policy:
  // b(a|s) = (1-ε)π(a|s) + ε/|A|.
  // This construction does not need qStar; qStar is only used to define the target policy and for RMSE validation.
  const behaviorPolicy: Policy = useMemo(() => {
    const numActions = 5;
    return targetPolicy.map((row) =>
      row.map((p) => (1 - epsilon) * p + epsilon / numActions)
    );
  }, [targetPolicy, epsilon]);

  function run() {
    const res = offPolicyMCEvaluation(targetPolicy, behaviorPolicy, config, episodesPerPair, type, horizonT);
    setResult(res);
  }

  const targetValues = useMemo(() => policyWeightedStateValues(qStar, targetPolicy), [qStar, targetPolicy]);
  const behaviorValues = useMemo(() => solveStateValues(behaviorPolicy, config), [behaviorPolicy, config]);
  const estimatedTargetValues = result
    ? policyWeightedStateValues(result.qValues, targetPolicy)
    : null;

  const rmse = useMemo(() => {
    if (!result) return null;
    return qTableRMSE(result.qValues, qStar);
  }, [result, qStar]);

  const chartData = useMemo(() => {
    if (!result) return [];
    return result.qHistory.map((q, i) => ({
      pair: i,
      rmse: qTableRMSE(q, qStar),
    }));
  }, [result, qStar]);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Scale className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          5.2 异策略蒙特卡洛与重要性采样
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          用行为策略 b 采样数据，通过重要性采样估计目标策略 π 的动作值。这是模型-free 策略评估的强大工具。
        </p>
        <p className="text-gray-500 text-sm max-w-2xl mx-auto mt-2">
          本页面预先使用模型计算 q*，仅用于构造一个可控的目标策略和验证估计误差。普通/加权重要性采样的估计过程本身没有访问环境模型。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="重要性采样比率"
          formula={
            <KaTeX
              math={String.raw`\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(a_k|s_k)}{b(a_k|s_k)}`}
              display
            />
          }
          description="把行为策略下得到的回报按目标策略与行为策略的概率比进行加权。"
        />
        <FormulaCard
          title="普通重要性采样"
          formula={
            <KaTeX
              math={String.raw`q(s,a) = \frac{\sum_{i=1}^{N} \rho_i G_i}{N}`}
              display
            />
          }
          description="直接对加权回报取平均。期望无偏，但方差可能很大。"
        />
        <FormulaCard
          title="加权重要性采样"
          formula={
            <KaTeX
              math={String.raw`q(s,a) = \frac{\sum_{i=1}^{N} \rho_i G_i}{\sum_{i=1}^{N} \rho_i}`}
              display
            />
          }
          description="用权重之和归一化。估计有偏但方差更小，实践中更稳定。"
        />
      </section>

      <InteractiveDemo title="目标策略 vs 行为策略">
        <div className="grid md:grid-cols-2 gap-6">
          <div className="flex flex-col items-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">目标策略 π（确定性贪心）与 V_π</h3>
            <GridWorld config={config} policy={targetPolicy} values={targetValues} showValues className="max-w-full" />
          </div>
          <div className="flex flex-col items-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">行为策略 b（ε-soft）与 V_b</h3>
            <GridWorld config={config} policy={behaviorPolicy} values={behaviorValues} showValues className="max-w-full" />
          </div>
        </div>
      </InteractiveDemo>

      <InteractiveDemo title="异策略 MC 估计">
        <div className="grid lg:grid-cols-[1fr_340px] gap-6">
          <div className="flex flex-col gap-4">
            {result && (
              <>
                <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
                  <GridWorld
                    config={config}
                    policy={greedyPolicy(result.qValues)}
                    values={estimatedTargetValues ?? undefined}
                    showValues
                    className="max-w-full"
                  />
                  <p className="mt-3 text-sm text-gray-500 text-center">
                    估计得到的贪心策略（基于 {type === 'ordinary' ? '普通' : '加权'} IS）
                  </p>
                </div>
                <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
                  <LineChart
                    data={chartData}
                    xKey="pair"
                    xLabel="已更新的 (s,a) 对数"
                    yLabel="RMSE(q, q*)"
                    series={[{ key: 'rmse', name: 'RMSE', color: '#2563eb' }]}
                    height={220}
                  />
                </div>
              </>
            )}
            {!result && (
              <div className="flex items-center justify-center h-64 bg-gray-50 rounded-xl border border-gray-200 text-gray-500">
                设置参数后点击运行，查看异策略 MC 估计结果
              </div>
            )}
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">设置</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm text-gray-700 mb-1">
                    <span>行为策略 ε</span>
                    <span className="font-mono">{epsilon.toFixed(2)}</span>
                  </div>
                  <Slider value={[epsilon]} min={0.05} max={1} step={0.05} onValueChange={([v]) => setEpsilon(v)} />
                  <p className="text-xs text-gray-500 mt-1">必须 ε &gt; 0，才能保证行为策略覆盖目标策略。</p>
                </div>
                <div>
                  <div className="flex justify-between text-sm text-gray-700 mb-1">
                    <span>每对 (s,a) 采样回合数</span>
                    <span className="font-mono">{episodesPerPair}</span>
                  </div>
                  <Slider value={[episodesPerPair]} min={1} max={100} step={1} onValueChange={([v]) => setEpisodesPerPair(v)} />
                </div>
                <div>
                  <label className="text-sm text-gray-700 block mb-1">轨迹长度 T</label>
                  <Select value={String(horizonT)} onValueChange={(v) => setHorizonT(Number(v))}>
                    <SelectTrigger><SelectValue/></SelectTrigger>
                    <SelectContent>
                      {[10, 20, 30, 50, 100, 200].map((t) => (
                        <SelectItem key={t} value={String(t)}>T={t}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-gray-500 mt-1">当前为 continuing task，使用长度为 T 的截断轨迹。RMSE 同时包含有限样本误差和截断误差。</p>
                </div>
                <div>
                  <label className="text-sm text-gray-700 block mb-1">估计方式</label>
                  <Select value={type} onValueChange={(v) => setType(v as EstimateType)}>
                    <SelectTrigger>
                      <SelectValue placeholder="选择估计方式" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ordinary">普通重要性采样</SelectItem>
                      <SelectItem value="weighted">加权重要性采样</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button size="sm" onClick={run} className="w-full bg-blue-600 hover:bg-blue-700">
                  运行异策略 MC
                </Button>
              </CardContent>
            </Card>

            {rmse !== null && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">结果</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-gray-700 space-y-1">
                  <div>RMSE（相对 q*）：<span className="font-mono font-semibold">{rmse.toFixed(4)}</span></div>
                  <div>总样本数：{(config.rows * config.cols * 5 * episodesPerPair).toLocaleString()}</div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </InteractiveDemo>

      {result && result.lastTrajectory.length > 0 && (
        <InteractiveDemo title="重要性采样示例：第一条 (s0,a0) 轨迹">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-gray-500 border-b">
                <tr>
                  <th className="py-2">步 t</th>
                  <th>状态 s_t</th>
                  <th>动作 a_t</th>
                  <th>奖励 r_{'{t+1}'}</th>
                  <th>π(a|s)</th>
                  <th>b(a|s)</th>
                  <th>单步比 ρ_t</th>
                  <th>累积 ρ_{'{1:t}'}</th>
                </tr>
              </thead>
              <tbody className="text-gray-700">
                {result.lastTrajectory.map((step, idx) => {
                  const piProb = targetPolicy[step.state][step.action];
                  const bProb = behaviorPolicy[step.state][step.action];
                  const ratio = bProb === 0 ? 0 : piProb / bProb;
                  let cum = 1;
                  for (let i = 1; i <= idx; i++) {
                    const s = result.lastTrajectory[i].state;
                    const a = result.lastTrajectory[i].action;
                    const bp = behaviorPolicy[s][a];
                    cum *= bp === 0 ? 0 : targetPolicy[s][a] / bp;
                  }
                  return (
                    <tr key={idx} className="border-b last:border-0">
                      <td className="py-2">{idx}</td>
                      <td>s{step.state + 1}</td>
                      <td>{ACTION_NAMES[step.action]}</td>
                      <td>{step.reward.toFixed(2)}</td>
                      <td>{piProb.toFixed(3)}</td>
                      <td>{bProb.toFixed(3)}</td>
                      <td>{idx === 0 ? '\u2014' : ratio.toFixed(3)}</td>
                      <td className='font-mono'>{idx === 0 ? '1.000' : cum.toFixed(3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className='mt-3 space-y-2'>
            <p className='text-sm text-gray-600'>
              <strong>第 0 步：</strong>初始动作 a_0 已条件化（由外部指定），不计入重要性权重，ρ = 1。
            </p>
            <p className='text-sm text-gray-600'>
              <strong>第 1 步起：</strong>累积权重从第 1 步开始计算 ρ_{'{1:t}'} = ∏_{'{k=1}'}^t π(a_k|s_k) / b(a_k|s_k)。
            </p>
            <p className='text-sm text-gray-600'>
              对 (s0,a0) 的重要性采样比率为上表最后一行的累积 ρ，回报 G 乘以该比率即得到目标策略下的加权回报。长度为 1 的轨迹，其 importance ratio 等于 1。
            </p>
          </div>
        </InteractiveDemo>
      )}

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">小结与 Q&A</h2>
        <ConceptAccordion
          items={[
            {
              id: 'summary',
              title: '本章小结',
              content: (
                <ul className="list-disc pl-5 space-y-2">
                  <li>同策略（on-policy）用当前策略本身采样；异策略（off-policy）用另一个行为策略采样。</li>
                  <li>重要性采样通过概率比 ρ 把行为策略样本“转换”为目标策略样本。</li>
                  <li>普通 IS 无偏但方差大；加权 IS 有偏但方差小，更实用。</li>
                  <li>当初始 (s,a) 已条件化时，重要性比率从下一动作开始计算（ρ_{'{1:t}'}），初始动作不纳入比率。</li>
                  <li>行为策略必须覆盖目标策略：b(a|s) &gt; 0  whenever π(a|s) &gt; 0。</li>
                </ul>
              ),
            },
            {
              id: 'qa1',
              title: 'Q: 为什么行为策略需要覆盖目标策略？',
              content:
                '如果目标策略会在某个状态选择动作 a，但行为策略选择 a 的概率为 0，那么我们就永远采不到该动作下的轨迹，无法估计 q_π(s,a)。',
            },
            {
              id: 'qa1b',
              title: 'Q: 为什么初始动作不计入重要性权重？',
              content:
                '在估计 q_π(s0,a0) 时，初始动作 a0 是外部条件给定的——无论目标策略还是行为策略，都以相同条件产生 a0。因此 a0 的概率比恒为 1，不需要纳入重要性比率。比率只从 a1 开始计算。',
            },
            {
              id: 'qa2',
              title: 'Q: 普通 IS 和加权 IS 哪个更好？',
              content:
                '普通 IS 理论上无偏，但当 ρ 很大时方差爆炸；加权 IS 用权重归一化，方差更小，且随着样本增加偏差逐渐消失。实际中常用加权 IS。',
            },
          ]}
        />
      </section>
    </div>
  );
}
