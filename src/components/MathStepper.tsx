import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import KaTeX from './KaTeX';

interface Step {
  title: string;
  math: string;
  explanation: string;
}

interface MathStepperProps {
  steps: Step[];
  className?: string;
}

export default function MathStepper({ steps, className = '' }: MathStepperProps) {
  const [current, setCurrent] = useState(0);

  function prev() {
    setCurrent((c) => Math.max(0, c - 1));
  }

  function next() {
    setCurrent((c) => Math.min(steps.length - 1, c + 1));
  }

  const step = steps[current];

  return (
    <div className={`bg-white border border-blue-200 rounded-xl overflow-hidden ${className}`}>
      <div className="bg-blue-50 px-4 py-3 border-b border-blue-100 flex items-center justify-between">
        <div className="text-sm font-medium text-blue-800">
          步骤 {current + 1} / {steps.length}: {step.title}
        </div>
        <div className="flex gap-1">
          <Button variant="outline" size="sm" onClick={prev} disabled={current === 0}>
            <ChevronLeft className="w-4 h-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={next} disabled={current === steps.length - 1}>
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>
      </div>
      <div className="p-5 space-y-4">
        <div className="flex justify-center">
          <KaTeX math={step.math} display />
        </div>
        <p className="text-sm text-gray-600 leading-relaxed">{step.explanation}</p>
      </div>
      <div className="flex gap-1 px-4 pb-4">
        {steps.map((_, i) => (
          <button
            key={i}
            onClick={() => setCurrent(i)}
            className={`h-1.5 flex-1 rounded-full transition-colors ${
              i <= current ? 'bg-blue-600' : 'bg-gray-200'
            }`}
          />
        ))}
      </div>
    </div>
  );
}
