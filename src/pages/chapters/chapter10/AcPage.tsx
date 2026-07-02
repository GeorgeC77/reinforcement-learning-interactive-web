import { useState, useMemo } from 'react';
import { Activity, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import GridWorld from '@/components/rl/GridWorld';
import { DEFAULT_CONFIG, actorCritic } from '@/lib/rl/gridworld';

export default function Chapter10AcPage() {
  const [actorAlpha, setActorAlpha] = useState(0.05);
  const [criticAlpha, setCriticAlpha] = useState(0.1);
  const [episodes, setEpisodes] = useState(150);
  const [step, setStep] = useState(0);

  const config = DEFAULT_CONFIG;

  const history = useMemo(() => {
    const { values, policies, rewardHistory } = actorCritic(
      config,
      actorAlpha,
      criticAlpha,
      episodes
    );
    return values.map((v, i) => ({
      values: v,
      policy: policies[i],
      reward: rewardHistory[i],
    }));
  }, [config, actorAlpha, criticAlpha, episodes]);

  const current = history[Math.min(step, history.length - 1)];
  const maxStep = history.length - 1;

  function next() {
    setStep((s) => Math.min(s + 1, maxStep));
  }
  function prev() {
    setStep((s) => Math.max(s - 1, 0));
  }
  function reset() {
    setStep(0);
  }

  const avgReward =
    history.slice(1, step + 1).reduce((sum, h) => sum + h.reward, 0) /
    Math.max(1, step);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Activity className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          10.1 Actor-Critic
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          在 GridWorld 上同时训练策略（Actor）和值函数（Critic），观察 TD 误差如何驱动两者共同进步。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="TD 误差"
          formula={<KaTeX math={String.raw`\delta_t = r_{t+1} + \gamma v(s_{t+1}) - v(s_t)`} display />}
          description="Critic 用 TD 误差更新状态值，Actor 用同一误差更新策略。"
        />
        <FormulaCard
          title="Critic 更新"
          formula={<KaTeX math={String.raw`v(s_t) \leftarrow v(s_t) + \alpha_w \delta_t`} display />}
          description="沿 TD 误差方向修正当前状态的值估计。"
        />
        <FormulaCard
          title="Actor 更新"
          formula={
            <KaTeX
              math={String.raw`\theta \leftarrow \theta + \alpha_\theta \delta_t \nabla_\theta \log \pi_\theta(a_t|s_t)`}
              display
            />
          }
          description="若 TD 误差为正，增加该动作的概率；若为负，则减小。"
        />
      </section>

      <InteractiveDemo title="Actor-Critic 训练回放">
        <div className="grid lg:grid-cols-[1fr_340px] gap-6">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <GridWorld
              config={config}
              policy={current.policy}
              values={current.values}
              showValues
              className="max-w-full"
            />
            <p className="mt-4 text-sm text-gray-500 text-center">
              第 {step} 回合后的 Actor 策略与 Critic 状态值
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">超参数</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="text-sm text-gray-600 mb-1">Actor 学习率 α_θ</div>
                  <Slider value={[actorAlpha]} min={0.001} max={0.2} step={0.001} onValueChange={([v]) => setActorAlpha(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{actorAlpha.toFixed(3)}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">Critic 学习率 α_w</div>
                  <Slider value={[criticAlpha]} min={0.001} max={0.5} step={0.001} onValueChange={([v]) => setCriticAlpha(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{criticAlpha.toFixed(3)}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">训练回合数</div>
                  <Slider value={[episodes]} min={20} max={300} step={10} onValueChange={([v]) => setEpisodes(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{episodes}</div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">训练统计</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-gray-700">
                <div>当前回合：<span className="font-mono font-semibold">{step} / {maxStep}</span></div>
                <div>本回合回报：<span className="font-mono">{current.reward.toFixed(2)}</span></div>
                <div>平均回报（前 {step} 回合）：<span className="font-mono">{avgReward.toFixed(2)}</span></div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">回放控制</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={prev} disabled={step === 0} className="flex-1">上一步</Button>
                  <Button size="sm" onClick={next} disabled={step === maxStep} className="flex-1">下一步</Button>
                </div>
                <Button variant="outline" size="sm" onClick={reset} className="w-full">重置</Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </InteractiveDemo>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">观察与思考</h2>
        <ul className="space-y-3 text-gray-700">
          <li><strong>协同训练：</strong>Critic 提供更稳定的 TD 误差信号，Actor 据此调整动作概率。</li>
          <li><strong>单步更新：</strong>与 REINFORCE 不同，Actor-Critic 每步都能更新，样本效率更高。</li>
          <li><strong>学习率平衡：</strong>Critic 学习率过大可能导致值估计震荡，进而误导 Actor。</li>
        </ul>
      </section>
    </div>
  );
}
