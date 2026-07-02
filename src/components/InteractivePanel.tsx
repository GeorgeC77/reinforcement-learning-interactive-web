import type { ReactNode } from 'react';

interface InteractivePanelProps {
  chart: ReactNode;
  controls: ReactNode;
  hint?: string;
  className?: string;
}

export default function InteractivePanel({
  chart,
  controls,
  hint,
  className = '',
}: InteractivePanelProps) {
  return (
    <div className={`w-full ${className}`}>
      {hint && (
        <p className="text-sm text-med-gray mb-3 font-sans">{hint}</p>
      )}
      <div className="flex flex-col lg:flex-row gap-4">
        <div className="w-full lg:w-[65%]">
          <div className="bg-white border border-border-gray rounded-xl p-4 overflow-hidden">
            {chart}
          </div>
        </div>
        <div className="w-full lg:w-[35%]">
          <div className="bg-white border border-border-gray rounded-xl p-5">
            {controls}
          </div>
        </div>
      </div>
    </div>
  );
}
