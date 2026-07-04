import GridWorld from './rl/GridWorld';
import type { GridWorldConfig, Policy, StateValues } from '@/lib/rl/gridworld';

interface PolicyComparisonProps {
  config: GridWorldConfig;
  leftTitle: string;
  rightTitle: string;
  leftPolicy: Policy;
  rightPolicy: Policy;
  leftValues?: StateValues;
  rightValues?: StateValues;
  className?: string;
}

export default function PolicyComparison({
  config,
  leftTitle,
  rightTitle,
  leftPolicy,
  rightPolicy,
  leftValues,
  rightValues,
  className,
}: PolicyComparisonProps) {
  return (
    <div className={`grid md:grid-cols-2 gap-4 ${className ?? ''}`}>
      <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
        <div className="text-center text-sm font-medium text-gray-700 mb-3">{leftTitle}</div>
        <div className="flex justify-center">
          <GridWorld
            config={config}
            policy={leftPolicy}
            values={leftValues}
            showValues
            className="max-w-full"
          />
        </div>
      </div>
      <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
        <div className="text-center text-sm font-medium text-gray-700 mb-3">{rightTitle}</div>
        <div className="flex justify-center">
          <GridWorld
            config={config}
            policy={rightPolicy}
            values={rightValues}
            showValues
            className="max-w-full"
          />
        </div>
      </div>
    </div>
  );
}
