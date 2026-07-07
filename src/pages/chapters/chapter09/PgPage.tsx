import { useState, useMemo, useEffect } from 'react';
import { GitBranch, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
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
import AlgorithmPlayer from '@/components/AlgorithmPlayer';
import LineChart from '@/components/LineChart';
import ConceptAccordion from '@/components/ConceptAccordion';
import {
  DEFAULT_CONFIG,
  EPISODIC_PATH_CONFIG,
  ACTION_NAMES,
  type Action,
  reinforceBandit,
  reinforceWithBaseline,
  reinforceMDP,
  reinforceMDPWithBaseline,
} from '@/lib/rl/gridworld';

type TabKey = 'bandit' | 'policy' | 'mdp' | 'baseline';

export default function Chapter09PgPage() {
  const [tab, setTab] = useState<TabKey>('bandit');

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <GitBranch className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          第 9 章 策略梯度方法
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          从 bandit 到完整 MDP：策略梯度定理、softmax 策略表示、REINFORCE 及其带基线变体。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="策略梯度定理"
          formula={
            <KaTeX
              math={String.raw`\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot q_{\pi_\theta}(s,a) \right]`}
              display
            />
          }
          description="策略目标对参数的梯度可写成 log-策略梯度与动作值的期望。"
        />
        <FormulaCard
          title="Softmax 策略"
          formula={
            <KaTeX
              math={String.raw`\pi(a|s,\theta) = \frac{e^{\theta(s,a)}}{\sum_{a'} e^{\theta(s,a')}}`}
              display
            />
          }
          description="每个状态都有一组动作偏好 θ(s,a)，softmax 把它们转成概率分布。"
        />
        <FormulaCard
          title="REINFORCE 更新"
          formula={
            <KaTeX
              math={String.raw`\theta(s,a) \leftarrow \theta(s,a) + \alpha G_t \bigl( \mathbf{1}_{a=A_t} - \pi(a|s,\theta) \bigr)`}
              display
            />
          }
          description="用完整回合的回报 G_t 加权 log-策略梯度，提升高回报动作的概率。"
        />
        <FormulaCard
          title="带基线的 REINFORCE"
          formula={
            <KaTeX
              math={String.raw`\theta(s,a) \leftarrow \theta(s,a) + \alpha (G_t - b(s)) \bigl( \mathbf{1}_{a=A_t} - \pi(a|s,\theta) \bigr)`}
              display
            />
          }
          description="用状态基线 b(s) 降低方差，不改变期望梯度。"
        />
      </section>

      <Tabs value={tab} onValueChange={(v) => setTab(v as TabKey)}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="bandit">Bandit 入门</TabsTrigger>
          <TabsTrigger value="policy">策略表示</TabsTrigger>
          <TabsTrigger value="mdp">GridWorld REINFORCE</TabsTrigger>
          <TabsTrigger value="baseline">Baseline 降方差</TabsTrigger>
        </TabsList>

        <TabsContent value="bandit" className="mt-4">
          <BanditDemo />
        </TabsContent>
        <TabsContent value="policy" className="mt-4">
          <PolicyRepresentationDemo />
        </TabsContent>
        <TabsContent value="mdp" className="mt-4">
          <MDPDemo />
        </TabsContent>
        <TabsContent value="baseline" className="mt-4">
          <BaselineDemo />
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
                  <li>策略梯度直接优化策略参数，适用于连续或大动作空间。</li>
                  <li>softmax 策略把动作偏好转换成概率分布。</li>
                  <li>REINFORCE 用蒙特卡洛回报估计梯度，方差大。</li>
                  <li>在完整 MDP 中，回报 G_t 是从时刻 t 开始的折扣累计奖励。</li>
                  <li>引入基线可显著降低方差，且不影响梯度期望。</li>
                </ul>
              ),
            },
            {
              id: 'qa1',
              title: 'Q: 为什么基线能降低方差但不改变期望？',
              content:
                '因为 Σ_a ∇log π(a|s) π(a|s) b(s) = b(s) ∇Σ_a π(a|s) = b(s) ∇1 = 0，所以减去基线后期望梯度不变，但估计量的方差减小。',
            },
            {
              id: 'qa2',
              title: 'Q: REINFORCE 与 Actor-Critic 的主要区别？',
              content:
                'REINFORCE 用一个完整 episode 的回报 G_t 作为更新信号；Actor-Critic 则用 critic 估计的 TD error，可以每步更新。',
            },
          ]}
        />
      </section>
    </div>
  );
}

// ------------------- Bandit Demo -------------------
function BanditDemo() {
  const [seed, setSeed] = useState(0);
  const [alpha, setAlpha] = useState(0.2);
  const [beta, setBeta] = useState(0.1);
  const [episodes, setEpisodes] = useState(200);

  const actionRewards = [1, 3];
  const initialTheta = [0, 0];

  const reinforceData = useMemo(() => {
    void seed;
    const { policyHistory, rewardHistory } = reinforceBandit(actionRewards, initialTheta, alpha, episodes);
    return policyHistory.map((policy, i) => ({
      episode: i,
      action0: policy[0],
      action1: policy[1],
      reward: rewardHistory[i - 1] ?? 0,
    }));
  }, [alpha, episodes, seed]);

  const baselineData = useMemo(() => {
    void seed;
    const { policyHistory, rewardHistory, baselineHistory } = reinforceWithBaseline(
      actionRewards,
      initialTheta,
      alpha,
      beta,
      episodes
    );
    return policyHistory.map((policy, i) => ({
      episode: i,
      action0: policy[0],
      action1: policy[1],
      baseline: baselineHistory[i],
      reward: rewardHistory[i - 1] ?? 0,
    }));
  }, [alpha, beta, episodes, seed]);

  const finalPolicy = reinforceData[reinforceData.length - 1];

  return (
    <div className="space-y-6">
      <InteractiveDemo title="Bandit REINFORCE：动作偏好到选择概率">
        <div className="grid lg:grid-cols-[1fr_340px] gap-6">
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <LineChart
              data={reinforceData}
              xKey="episode"
              xLabel="回合"
              yLabel="选择概率"
              series={[
                { key: 'action0', name: '动作 0 (r≈1)', color: '#2563eb' },
                { key: 'action1', name: '动作 1 (r≈3)', color: '#ef4444' },
              ]}
            />
          </div>
          <div className="space-y-4">
            <ParamCard>
              <Param label="α" value={alpha} set={setAlpha} min={0.01} max={0.5} step={0.01} fixed={2} />
              <Param label="回合数" value={episodes} set={setEpisodes} min={20} max={500} step={10} />
            </ParamCard>
            <StrategyCard policy={finalPolicy} />
            <Button onClick={() => setSeed((s) => s + 1)} variant="outline" className="w-full">重新采样</Button>
          </div>
        </div>
      </InteractiveDemo>

      <InteractiveDemo title="Bandit REINFORCE + Baseline">
        <div className="grid lg:grid-cols-[1fr_340px] gap-6">
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <LineChart
              data={baselineData}
              xKey="episode"
              xLabel="回合"
              yLabel="概率 / 基线"
              series={[
                { key: 'action0', name: '动作 0', color: '#2563eb' },
                { key: 'action1', name: '动作 1', color: '#ef4444' },
                { key: 'baseline', name: '基线 b', color: '#22c55e', strokeDasharray: '4 2' },
              ]}
            />
          </div>
          <div className="space-y-4">
            <ParamCard>
              <Param label="α" value={alpha} set={setAlpha} min={0.01} max={0.5} step={0.01} fixed={2} />
              <Param label="β（基线更新）" value={beta} set={setBeta} min={0.001} max={0.5} step={0.001} fixed={3} />
              <Param label="回合数" value={episodes} set={setEpisodes} min={20} max={500} step={10} />
            </ParamCard>
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-base">基线作用</CardTitle></CardHeader>
              <CardContent className="text-sm text-gray-700 space-y-2">
                <p>回报 G 大于基线时，提升该动作概率；小于基线时，降低概率。</p>
                <p>基线只改变梯度估计的方差，不改变期望，因此策略梯度定理仍然成立。</p>
              </CardContent>
            </Card>
            <Button onClick={() => setSeed((s) => s + 1)} variant="outline" className="w-full">重新采样</Button>
          </div>
        </div>
      </InteractiveDemo>
    </div>
  );
}

// ------------------- Policy Representation Demo -------------------
function PolicyRepresentationDemo() {
  const config = DEFAULT_CONFIG;
  const [selectedState, setSelectedState] = useState(0);
  const [preferences, setPreferences] = useState<number[]>(new Array(5).fill(0));

  const policy = useMemo(() => {
    const maxPref = Math.max(...preferences);
    const exps = preferences.map((p) => Math.exp(p - maxPref));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }, [preferences]);

  return (
    <InteractiveDemo title="策略参数化：pi(a|s,theta) = softmax(theta(s,a))">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
          <div className="text-sm text-gray-600 mb-3">选择状态 s{selectedState + 1} 的动作偏好 θ</div>
          <div className="grid grid-cols-5 gap-2 mb-4">
            {preferences.map((p, a) => (
              <div key={a} className="flex flex-col items-center">
                <span className="text-xs text-gray-500 mb-1">{ACTION_NAMES[a]}</span>
                <input
                  type="number"
                  value={p.toFixed(1)}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value) || 0;
                    setPreferences((prev) => prev.map((x, i) => (i === a ? v : x)));
                  }}
                  className="w-16 px-1 py-1 text-center border rounded text-sm"
                />
              </div>
            ))}
          </div>
          <div className="w-full max-w-md space-y-2">
            {policy.map((prob, a) => (
              <div key={a} className="flex items-center gap-2 text-sm">
                <span className="w-12">{ACTION_NAMES[a]}</span>
                <div className="flex-1 h-4 bg-gray-200 rounded overflow-hidden">
                  <div className="h-full bg-blue-500" style={{ width: `${prob * 100}%` }} />
                </div>
                <span className="w-14 text-right font-mono">{prob.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">状态选择</CardTitle></CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-2">
                {Array.from({ length: config.rows * config.cols }, (_, i) => i).map((s) => (
                  <Button
                    key={s}
                    variant={selectedState === s ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => {
                      setSelectedState(s);
                      setPreferences(new Array(5).fill(0));
                    }}
                  >
                    s{s + 1}
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">观察</CardTitle></CardHeader>
            <CardContent className="text-sm text-gray-700 space-y-2">
              <p>softmax 把所有动作偏好映射到概率单纯形。</p>
              <p>给某个动作增加 θ 会提升它的概率，但也会压低其他动作的概率。</p>
              <p>在 REINFORCE 中，θ 不是手动设置，而是由梯度上升自动学习。</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- MDP REINFORCE Demo -------------------
function MDPDemo() {
  const [taskType, setTaskType] = useState<'continuing' | 'episodic'>('episodic');
  const config = useMemo(
    () => (taskType === 'episodic' ? EPISODIC_PATH_CONFIG : DEFAULT_CONFIG),
    [taskType]
  );
  const [alpha, setAlpha] = useState(0.05);
  const [episodes, setEpisodes] = useState(150);
  const [seed, setSeed] = useState(0);
  const [step, setStep] = useState(0);

  useEffect(() => {
    setStep(0);
  }, [taskType, config]);

  const result = useMemo(() => {
    void seed;
    return reinforceMDP(config, alpha, episodes, 30);
  }, [config, alpha, episodes, seed]);

  const maxStep = result.policyHistory.length - 1;
  const currentPolicy = result.policyHistory[Math.min(step, maxStep)];
  const currentEpisode = result.episodes[Math.min(step > 0 ? step - 1 : 0, result.episodes.length - 1)];

  const returnData = useMemo(
    () =>
      result.episodes.map((ep, i) => ({
        episode: i + 1,
        return: ep.totalReward,
      })),
    [result]
  );

  return (
    <InteractiveDemo title="GridWorld REINFORCE：完整 MDP 轨迹与参数更新">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld config={config} policy={currentPolicy} showValues={false} className="max-w-full" />
            <p className="mt-3 text-sm text-gray-500 text-center">第 {step} 回合后的 softmax 策略</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <LineChart
              data={returnData}
              xKey="episode"
              xLabel="回合"
              yLabel="回合总回报"
              series={[{ key: 'return', name: 'Episode Return', color: '#2563eb' }]}
              height={200}
            />
          </div>
          <EpisodeDetailsCard episode={currentEpisode} />
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">参数</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm text-gray-700 block mb-1">任务类型</label>
                <Select value={taskType} onValueChange={(v) => setTaskType(v as 'continuing' | 'episodic')}>
                  <SelectTrigger>
                    <SelectValue placeholder="选择任务类型" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="episodic">episodic path-finding（默认）</SelectItem>
                    <SelectItem value="continuing">教材 continuing GridWorld</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Param label="α" value={alpha} set={setAlpha} min={0.001} max={0.2} step={0.001} fixed={3} />
              <Param label="回合数" value={episodes} set={setEpisodes} min={20} max={300} step={10} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">回放控制</CardTitle></CardHeader>
            <CardContent>
              <AlgorithmPlayer maxStep={maxStep} currentStep={step} onStepChange={setStep} />
            </CardContent>
          </Card>
          <Button onClick={() => { setSeed((s) => s + 1); setStep(0); }} variant="outline" className="w-full">重新训练</Button>
        </div>
      </div>
    </InteractiveDemo>
  );
}

// ------------------- Baseline Demo -------------------
function BaselineDemo() {
  const [taskType, setTaskType] = useState<'continuing' | 'episodic'>('episodic');
  const config = useMemo(
    () => (taskType === 'episodic' ? EPISODIC_PATH_CONFIG : DEFAULT_CONFIG),
    [taskType]
  );
  const [alpha, setAlpha] = useState(0.05);
  const [beta, setBeta] = useState(0.1);
  const [episodes, setEpisodes] = useState(150);
  const [seed, setSeed] = useState(0);
  const [step, setStep] = useState(0);

  useEffect(() => {
    setStep(0);
  }, [taskType, config]);

  const result = useMemo(() => {
    void seed;
    return reinforceMDPWithBaseline(config, alpha, beta, episodes, 30);
  }, [config, alpha, beta, episodes, seed]);

  const maxStep = result.policyHistory.length - 1;
  const currentPolicy = result.policyHistory[Math.min(step, maxStep)];
  const currentEpisode = result.episodes[Math.min(step > 0 ? step - 1 : 0, result.episodes.length - 1)];

  const returnData = useMemo(
    () =>
      result.episodes.map((ep, i) => ({
        episode: i + 1,
        return: ep.totalReward,
      })),
    [result]
  );

  return (
    <InteractiveDemo title="带基线的 GridWorld REINFORCE：用 G_t - b(s) 降低方差">
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld config={config} policy={currentPolicy} showValues={false} className="max-w-full" />
            <p className="mt-3 text-sm text-gray-500 text-center">第 {step} 回合后的 softmax 策略</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
            <LineChart
              data={returnData}
              xKey="episode"
              xLabel="回合"
              yLabel="回合总回报"
              series={[{ key: 'return', name: 'Episode Return', color: '#ef4444' }]}
              height={200}
            />
          </div>
          <EpisodeDetailsCard episode={currentEpisode} showBaseline />
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">参数</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm text-gray-700 block mb-1">任务类型</label>
                <Select value={taskType} onValueChange={(v) => setTaskType(v as 'continuing' | 'episodic')}>
                  <SelectTrigger>
                    <SelectValue placeholder="选择任务类型" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="episodic">episodic path-finding（默认）</SelectItem>
                    <SelectItem value="continuing">教材 continuing GridWorld</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Param label="α" value={alpha} set={setAlpha} min={0.001} max={0.2} step={0.001} fixed={3} />
              <Param label="β（基线更新）" value={beta} set={setBeta} min={0.001} max={0.5} step={0.001} fixed={3} />
              <Param label="回合数" value={episodes} set={setEpisodes} min={20} max={300} step={10} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-base">回放控制</CardTitle></CardHeader>
            <CardContent>
              <AlgorithmPlayer maxStep={maxStep} currentStep={step} onStepChange={setStep} />
            </CardContent>
          </Card>
          <Button onClick={() => { setSeed((s) => s + 1); setStep(0); }} variant="outline" className="w-full">重新训练</Button>
        </div>
      </div>
    </InteractiveDemo>
  );
}

function EpisodeDetailsCard({
  episode,
  showBaseline = false,
}: {
  episode: { trajectory: { state: number; action: Action; reward: number; nextState: number }[]; totalReward: number; stepDetails: { state: number; action: Action; reward: number; prob: number; return: number }[] };
  showBaseline?: boolean;
}) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 text-sm">
      <h3 className="font-semibold text-gray-800 mb-2">本回合轨迹（总回报 {episode.totalReward.toFixed(2)}）</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead className="text-gray-500 border-b">
            <tr>
              <th className="py-1">t</th>
              <th>S_t</th>
              <th>A_t</th>
              <th>R_{'{t+1}'}</th>
              <th>G_t</th>
              <th>π(A_t|S_t)</th>
            </tr>
          </thead>
          <tbody className="text-gray-700">
            {episode.stepDetails.map((det, idx) => (
              <tr key={idx} className="border-b last:border-0">
                <td className="py-1">{idx}</td>
                <td>s{det.state + 1}</td>
                <td>{ACTION_NAMES[det.action]}</td>
                <td>{det.reward.toFixed(2)}</td>
                <td>{det.return.toFixed(2)}</td>
                <td className="font-mono">{det.prob.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {showBaseline && (
        <p className="mt-2 text-gray-600">基线 b(s) 是各状态回报的移动平均，更新量为 α·(G_t - b(s_t))·∇log π。</p>
      )}
    </div>
  );
}

function ParamCard({ children }: { children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader className="pb-2"><CardTitle className="text-base">参数</CardTitle></CardHeader>
      <CardContent className="space-y-3">{children}</CardContent>
    </Card>
  );
}

function Param({
  label,
  value,
  set,
  min,
  max,
  step,
  fixed,
}: {
  label: string;
  value: number;
  set: (v: number) => void;
  min: number;
  max: number;
  step: number;
  fixed?: number;
}) {
  return (
    <div>
      <div className="text-sm text-gray-600 mb-1">{label}</div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={([v]) => set(v)} />
      <div className="mt-1 text-center font-mono text-sm text-gray-700">
        {fixed !== undefined ? value.toFixed(fixed) : value}
      </div>
    </div>
  );
}

function StrategyCard({ policy }: { policy: { action0: number; action1: number } }) {
  return (
    <Card>
      <CardHeader className="pb-2"><CardTitle className="text-base">当前策略</CardTitle></CardHeader>
      <CardContent className="space-y-2 text-sm text-gray-700">
        <div>动作 0 概率：<span className="font-mono">{policy.action0.toFixed(3)}</span></div>
        <div>动作 1 概率：<span className="font-mono">{policy.action1.toFixed(3)}</span></div>
      </CardContent>
    </Card>
  );
}
