import { useState, useMemo } from 'react';
import { TrendingDown, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';
import { sgdQuadratic } from '@/lib/rl/gridworld';

export default function Chapter06SaPage() {
  const [wTrue, setWTrue] = useState(3);
  const [initialW, setInitialW] = useState(0);
  const [alpha, setAlpha] = useState(0.05);
  const [noiseStd, setNoiseStd] = useState(0.5);
  const [iterations, setIterations] = useState(100);
  const [seed, setSeed] = useState(0);

  const history = useMemo(() => {
    // Trigger recompute when seed changes
    void seed;
    return sgdQuadratic(wTrue, initialW, alpha, noiseStd, iterations);
  }, [wTrue, initialW, alpha, noiseStd, iterations, seed]);

  const width = 320;
  const height = 180;
  const padding = 24;
  const plotWidth = width - 2 * padding;
  const plotHeight = height - 2 * padding;

  const minW = Math.min(wTrue, initialW, ...history) - 1;
  const maxW = Math.max(wTrue, initialW, ...history) + 1;

  function scaleX(i: number) {
    return padding + (i / (history.length - 1 || 1)) * plotWidth;
  }
  function scaleY(w: number) {
    return padding + (1 - (w - minW) / (maxW - minW || 1)) * plotHeight;
  }

  const pathD = history
    .map((w, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i)} ${scaleY(w)}`)
    .join(' ');

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <TrendingDown className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          6.1 随机逼近与随机梯度下降
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          通过带噪声的梯度样本迭代优化参数，观察学习率与噪声对收敛的影响。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="Robbins-Monro 算法"
          formula={<KaTeX math={String.raw`w_{k+1} = w_k - \alpha_k \tilde{g}(w_k, \eta_k)`} display />}
          description="其中 g̃ 是对真实函数 g 的带噪声观测，用于迭代求 g(w)=0 的根。"
        />
        <FormulaCard
          title="随机梯度下降"
          formula={<KaTeX math={String.raw`w_{k+1} = w_k - \alpha_k \widetilde{\nabla} J(w_k)`} display />}
          description="用单个样本的梯度近似批量梯度，每一步计算量小，但带有随机噪声。"
        />
        <FormulaCard
          title="收敛条件"
          formula={<KaTeX math={String.raw`\sum_k \alpha_k = \infty, \quad \sum_k \alpha_k^2 < \infty`} display />}
          description="学习率要足够大以到达目标，又不能太大以致方差不收敛。"
        />
      </section>

      <InteractiveDemo title="SGD 优化二次函数">
        <div className="grid lg:grid-cols-[1fr_340px] gap-6">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <svg width={width} height={height} className="bg-white rounded-lg border border-gray-200">
              <line
                x1={padding}
                y1={scaleY(wTrue)}
                x2={width - padding}
                y2={scaleY(wTrue)}
                stroke="#22c55e"
                strokeDasharray="4 4"
                strokeWidth={1}
              />
              <path d={pathD} fill="none" stroke="#2563eb" strokeWidth={2} />
              <text x={width - padding} y={scaleY(wTrue) - 4} textAnchor="end" fontSize={10} fill="#22c55e">
                w* = {wTrue}
              </text>
              <text x={padding} y={height - 4} fontSize={10} fill="#6b7280">
                迭代次数
              </text>
              <text x={4} y={padding - 4} fontSize={10} fill="#6b7280">
                w
              </text>
            </svg>
            <p className="mt-4 text-sm text-gray-500 text-center">
              蓝色曲线为参数 w 的迭代轨迹，绿色虚线为真实最优值
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">参数设置</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="text-sm text-gray-600 mb-1">真实最优值 w*</div>
                  <Slider value={[wTrue]} min={-5} max={5} step={0.5} onValueChange={([v]) => setWTrue(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{wTrue}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">初始值 w₀</div>
                  <Slider value={[initialW]} min={-5} max={5} step={0.5} onValueChange={([v]) => setInitialW(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{initialW}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">学习率 α</div>
                  <Slider value={[alpha]} min={0.001} max={0.2} step={0.001} onValueChange={([v]) => setAlpha(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{alpha.toFixed(3)}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">梯度噪声标准差</div>
                  <Slider value={[noiseStd]} min={0} max={2} step={0.1} onValueChange={([v]) => setNoiseStd(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{noiseStd.toFixed(1)}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">迭代次数</div>
                  <Slider value={[iterations]} min={20} max={300} step={10} onValueChange={([v]) => setIterations(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{iterations}</div>
                </div>
              </CardContent>
            </Card>

            <Button onClick={() => setSeed((s) => s + 1)} variant="outline" className="w-full">
              重新采样随机梯度
            </Button>
          </div>
        </div>
      </InteractiveDemo>

      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">观察与思考</h2>
        <ul className="space-y-3 text-gray-700">
          <li><strong>学习率 α：</strong>α 太小收敛慢；α 太大会在最优值附近震荡甚至发散。</li>
          <li><strong>噪声：</strong>梯度噪声越大，收敛越不稳定，但合适的衰减学习率仍能收敛。</li>
          <li><strong>与 RL 的联系：</strong>TD 目标 r+γv(s') 就是对真实值的带噪声估计，因此 TD 也是一种随机逼近。</li>
        </ul>
      </section>
    </div>
  );
}
