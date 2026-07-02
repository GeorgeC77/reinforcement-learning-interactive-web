import { useState, useMemo } from 'react';
import { GitBranch, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import { reinforceBandit } from '@/lib/rl/gridworld';

export default function Chapter09PgPage() {
  const [alpha, setAlpha] = useState(0.2);
  const [episodes, setEpisodes] = useState(200);
  const [seed, setSeed] = useState(0);

  const actionRewards = [1, 3];
  const initialTheta = [0, 0];

  const { policyHistory, rewardHistory } = useMemo(() => {
    void seed;
    return reinforceBandit(actionRewards, initialTheta, alpha, episodes);
  }, [alpha, episodes, seed]);

  const width = 360;
  const height = 200;
  const padding = 28;
  const plotWidth = width - 2 * padding;
  const plotHeight = height - 2 * padding;

  function scaleX(i: number) {
    return padding + (i / (policyHistory.length - 1 || 1)) * plotWidth;
  }
  function scaleY(p: number) {
    return padding + (1 - p) * plotHeight;
  }

  const prob0Path = policyHistory
    .map((policy, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i)} ${scaleY(policy[0])}`)
    .join(' ');
  const prob1Path = policyHistory
    .map((policy, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i)} ${scaleY(policy[1])}`)
    .join(' ');

  const finalPolicy = policyHistory[policyHistory.length - 1];

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <GitBranch className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          9.1 策略梯度与 REINFORCE
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          在多臂老虎机上运行 REINFORCE，观察 softmax 策略如何倾向于选择期望奖励更高的动作。
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
          formula={<KaTeX math={String.raw`\pi(a) = \frac{e^{\theta_a}}{\sum_{a'} e^{\theta_{a'}}}`} display />}
          description="动作偏好 θ 越大，选择该动作的概率越高。"
        />
        <FormulaCard
          title="REINFORCE 更新"
          formula={<KaTeX math={String.raw`\theta_a \leftarrow \theta_a + \alpha \cdot G \cdot \bigl( \mathbb{I}(a) - \pi(a) \bigr)`} display />}
          description="采样得到的回报 G 越大，被采样动作的概率提升越多。"
        />
      </section>

      <InteractiveDemo title="REINFORCE 训练过程">
        <div className="grid lg:grid-cols-[1fr_340px] gap-6">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <svg width={width} height={height} className="bg-white rounded-lg border border-gray-200">
              <line x1={padding} y1={scaleY(0.5)} x2={width - padding} y2={scaleY(0.5)} stroke="#e5e7eb" strokeWidth={1} />
              <path d={prob0Path} fill="none" stroke="#2563eb" strokeWidth={2} />
              <path d={prob1Path} fill="none" stroke="#ef4444" strokeWidth={2} strokeDasharray="4 2" />
              <text x={width - padding} y={scaleY(finalPolicy[0]) - 4} textAnchor="end" fontSize={10} fill="#2563eb">
                动作 0 (r≈1)
              </text>
              <text x={width - padding} y={scaleY(finalPolicy[1]) + 12} textAnchor="end" fontSize={10} fill="#ef4444">
                动作 1 (r≈3)
              </text>
              <text x={padding} y={height - 4} fontSize={10} fill="#6b7280">回合数</text>
              <text x={4} y={padding - 4} fontSize={10} fill="#6b7280">概率</text>
            </svg>
            <p className="mt-4 text-sm text-gray-500 text-center">
              蓝色实线为动作 0 的选择概率，红色虚线为动作 1 的选择概率
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">参数设置</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="text-sm text-gray-600 mb-1">学习率 α</div>
                  <Slider value={[alpha]} min={0.01} max={0.5} step={0.01} onValueChange={([v]) => setAlpha(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{alpha.toFixed(2)}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">训练回合数</div>
                  <Slider value={[episodes]} min={20} max={500} step={10} onValueChange={([v]) => setEpisodes(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{episodes}</div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">当前策略</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-gray-700">
                <div>动作 0 概率：<span className="font-mono">{finalPolicy[0].toFixed(3)}</span></div>
                <div>动作 1 概率：<span className="font-mono">{finalPolicy[1].toFixed(3)}</span></div>
                <div>平均回报：<span className="font-mono">{(rewardHistory.reduce((a, b) => a + b, 0) / rewardHistory.length).toFixed(2)}</span></div>
              </CardContent>
            </Card>

            <Button onClick={() => setSeed((s) => s + 1)} variant="outline" className="w-full">
              重新采样训练
            </Button>
          </div>
        </div>
      </InteractiveDemo>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">观察与思考</h2>
        <ul className="space-y-3 text-gray-700">
          <li><strong>偏好高奖励动作：</strong>由于动作 1 的期望奖励更高，REINFORCE 会逐渐提高选择它的概率。</li>
          <li><strong>学习率影响：</strong>α 过大时策略概率可能震荡；α 过小则收敛缓慢。</li>
          <li><strong>方差问题：</strong>REINFORCE 使用完整回合回报，方差较大；后续可用基线或 Actor-Critic 降低方差。</li>
        </ul>
      </section>
    </div>
  );
}
