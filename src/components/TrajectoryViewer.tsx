import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';

export interface TrajectoryStep {
  t: number;
  state: number | string;
  action: number | string;
  reward: number;
  nextState: number | string;
}

interface TrajectoryViewerProps {
  steps: TrajectoryStep[];
  currentIndex?: number;
  highlightState?: number | string;
  className?: string;
}

export default function TrajectoryViewer({
  steps,
  currentIndex = -1,
  highlightState,
  className,
}: TrajectoryViewerProps) {
  return (
    <Card className={cn('overflow-hidden', className)}>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-3 py-2 text-left font-medium text-gray-600">t</th>
                <th className="px-3 py-2 text-left font-medium text-gray-600">s_t</th>
                <th className="px-3 py-2 text-left font-medium text-gray-600">a_t</th>
                <th className="px-3 py-2 text-left font-medium text-gray-600">r_t+1</th>
                <th className="px-3 py-2 text-left font-medium text-gray-600">s_t+1</th>
              </tr>
            </thead>
            <tbody>
              {steps.map((step, idx) => {
                const isActive = idx === currentIndex;
                const isHighlight =
                  highlightState !== undefined &&
                  (step.state === highlightState || step.nextState === highlightState);
                return (
                  <tr
                    key={idx}
                    className={cn(
                      'border-b border-gray-100 transition-colors',
                      isActive && 'bg-blue-50',
                      isHighlight && !isActive && 'bg-amber-50',
                      !isActive && !isHighlight && 'hover:bg-gray-50'
                    )}
                  >
                    <td className="px-3 py-2 font-mono text-gray-700">{step.t}</td>
                    <td className="px-3 py-2 font-mono text-gray-700">{step.state}</td>
                    <td className="px-3 py-2 font-mono text-gray-700">{step.action}</td>
                    <td className="px-3 py-2 font-mono text-gray-700">{step.reward.toFixed(2)}</td>
                    <td className="px-3 py-2 font-mono text-gray-700">{step.nextState}</td>
                  </tr>
                );
              })}
              {steps.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-3 py-4 text-center text-gray-500">
                    暂无轨迹数据
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
