import { useState, useMemo } from 'react';
import { Brain, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import KaTeX from '@/components/KaTeX';
import FormulaCard from '@/components/FormulaCard';
import InteractiveDemo from '@/components/InteractiveDemo';

function polynomialFeatures(x: number, degree: number): number[] {
  const phi = [1];
  for (let i = 1; i <= degree; i++) {
    phi.push(phi[i - 1] * x);
  }
  return phi;
}

function trueValue(x: number): number {
  return Math.sin(Math.PI * x);
}

export default function Chapter08FaPage() {
  const [degree, setDegree] = useState(3);
  const [alpha, setAlpha] = useState(0.05);
  const [iterations, setIterations] = useState(200);
  const [noiseStd, setNoiseStd] = useState(0.1);
  const [seed, setSeed] = useState(0);

  const { weights, predictions } = useMemo(() => {
    void seed;
    const numFeatures = degree + 1;
    let w = new Array(numFeatures).fill(0);
    const n = 50;

    for (let k = 0; k < iterations; k++) {
      // Sample a state uniformly from [-1, 1]
      const x = Math.random() * 2 - 1;
      const phi = polynomialFeatures(x, degree);
      const target = trueValue(x) + (Math.random() * 2 - 1) * noiseStd;
      const pred = phi.reduce((sum, f, i) => sum + f * w[i], 0);
      const error = target - pred;
      for (let i = 0; i < numFeatures; i++) {
        w[i] += alpha * error * phi[i];
      }
    }

    const xs = Array.from({ length: n }, (_, i) => -1 + (2 * i) / (n - 1));
    const preds = xs.map((x) => {
      const phi = polynomialFeatures(x, degree);
      return phi.reduce((sum, f, i) => sum + f * w[i], 0);
    });

    return { weights: w, predictions: { xs, preds } };
  }, [degree, alpha, iterations, noiseStd, seed]);

  const width = 360;
  const height = 220;
  const padding = 28;
  const plotWidth = width - 2 * padding;
  const plotHeight = height - 2 * padding;

  const minY = -1.5;
  const maxY = 1.5;

  function scaleX(x: number) {
    return padding + ((x + 1) / 2) * plotWidth;
  }
  function scaleY(y: number) {
    return padding + (1 - (y - minY) / (maxY - minY)) * plotHeight;
  }

  const truePathD = predictions.xs
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(x)} ${scaleY(trueValue(x))}`)
    .join(' ');

  const predPathD = predictions.xs
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(x)} ${scaleY(predictions.preds[i])}`)
    .join(' ');

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-12">
      <section className="text-center py-10 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center">
            <Brain className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
          8.1 值函数近似
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          用多项式特征线性逼近一维状态值函数，观察特征维度与学习率对拟合效果的影响。
        </p>
        <p className="mt-4 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用。
        </p>
      </section>

      <section className="space-y-4">
        <FormulaCard
          title="线性值函数近似"
          formula={<KaTeX math={String.raw`\hat{v}(s, w) = \phi(s)^\top w = \sum_{i=0}^d \phi_i(s) w_i`} display />}
          description="用状态特征 φ(s) 的线性组合来近似状态值，参数 w 决定了近似形状。"
        />
        <FormulaCard
          title="梯度下降更新"
          formula={<KaTeX math={String.raw`w \leftarrow w + \alpha \bigl[ U_t - \hat{v}(s_t, w) \bigr] \nabla_w \hat{v}(s_t, w)`} display />}
          description="目标 U_t 可以是 TD 目标或真实值，沿梯度方向减小预测误差。"
        />
        <FormulaCard
          title="多项式特征"
          formula={<KaTeX math={String.raw`\phi(s) = [1, s, s^2, \dots, s^d]^\top`} display />}
          description="特征维度 d 越高，模型表达能力越强，但也越容易过拟合。"
        />
      </section>

      <InteractiveDemo title="多项式逼近状态值函数">
        <div className="grid lg:grid-cols-[1fr_340px] gap-6">
          <div className="flex flex-col items-center justify-center bg-gray-50 rounded-xl p-6 border border-gray-200">
            <svg width={width} height={height} className="bg-white rounded-lg border border-gray-200">
              <line x1={padding} y1={scaleY(0)} x2={width - padding} y2={scaleY(0)} stroke="#e5e7eb" strokeWidth={1} />
              <path d={truePathD} fill="none" stroke="#22c55e" strokeWidth={2} />
              <path d={predPathD} fill="none" stroke="#2563eb" strokeWidth={2} strokeDasharray="4 2" />
              <text x={width - padding} y={scaleY(1.3)} textAnchor="end" fontSize={10} fill="#22c55e">
                真实值 v*(s)
              </text>
              <text x={width - padding} y={scaleY(1.0)} textAnchor="end" fontSize={10} fill="#2563eb">
                近似值 v̂(s,w)
              </text>
              <text x={padding} y={height - 4} fontSize={10} fill="#6b7280">状态 s</text>
              <text x={4} y={padding - 4} fontSize={10} fill="#6b7280">v</text>
            </svg>
            <p className="mt-4 text-sm text-gray-500 text-center">
              绿色实线为真实值函数，蓝色虚线为线性近似结果
            </p>
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">模型设置</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="text-sm text-gray-600 mb-1">多项式阶数 d</div>
                  <Slider value={[degree]} min={0} max={8} step={1} onValueChange={([v]) => setDegree(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{degree}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">学习率 α</div>
                  <Slider value={[alpha]} min={0.001} max={0.2} step={0.001} onValueChange={([v]) => setAlpha(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{alpha.toFixed(3)}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">训练迭代次数</div>
                  <Slider value={[iterations]} min={0} max={1000} step={50} onValueChange={([v]) => setIterations(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{iterations}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">目标噪声</div>
                  <Slider value={[noiseStd]} min={0} max={0.5} step={0.05} onValueChange={([v]) => setNoiseStd(v)} />
                  <div className="mt-1 text-center font-mono text-sm text-gray-700">{noiseStd.toFixed(2)}</div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">参数向量 w</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="font-mono text-sm text-gray-700 break-all">
                  [{weights.map((wi) => wi.toFixed(2)).join(', ')}]
                </div>
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
          <li><strong>特征维度：</strong>阶数 d 太低时无法捕捉真实函数的弯曲；d 太高时可能过拟合噪声。</li>
          <li><strong>学习率：</strong>α 过大导致参数震荡，α 过小收敛很慢。</li>
          <li><strong>从线性到非线性：</strong>当特征难以设计时，可用神经网络自动学习 φ(s)，这就是深度强化学习的核心思想。</li>
        </ul>
      </section>
    </div>
  );
}
