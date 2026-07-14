import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Play, Pause, SkipBack, SkipForward, RotateCcw } from 'lucide-react';
import { Slider } from '@/components/ui/slider';

interface AlgorithmPlayerProps {
  maxStep: number;
  currentStep?: number;
  onStepChange?: (step: number) => void;
  labels?: {
    step?: string;
    play?: string;
    pause?: string;
    reset?: string;
  };
  speeds?: number[];
  className?: string;
}

export default function AlgorithmPlayer({
  maxStep,
  currentStep: controlledStep,
  onStepChange,
  labels = { step: '步数', play: '播放', pause: '暂停', reset: '重置' },
  speeds = [0.5, 1, 2, 4],
  className = '',
}: AlgorithmPlayerProps) {
  const isControlled = controlledStep !== undefined;
  const [internalStep, setInternalStep] = useState(0);
  const currentStep = isControlled ? controlledStep : internalStep;

  const [isPlaying, setIsPlaying] = useState(false);
  const [speedIndex, setSpeedIndex] = useState(1);
  const stepRef = useRef(currentStep);
  stepRef.current = currentStep;

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const speed = speeds[speedIndex] ?? 1;
  const intervalMs = Math.max(50, Math.round(500 / speed));

  function setStep(step: number) {
    const clamped = Math.max(0, Math.min(maxStep, step));
    if (!isControlled) {
      setInternalStep(clamped);
    }
    onStepChange?.(clamped);
  }

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        const next = stepRef.current + 1;
        if (next >= maxStep) {
          setStep(maxStep);
          setIsPlaying(false);
        } else {
          setStep(next);
        }
      }, intervalMs);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, intervalMs, maxStep]);

  function togglePlay() {
    if (currentStep >= maxStep) {
      setStep(0);
      setIsPlaying(true);
    } else {
      setIsPlaying((p) => !p);
    }
  }

  function prev() {
    setIsPlaying(false);
    setStep(currentStep - 1);
  }

  function next() {
    setIsPlaying(false);
    setStep(currentStep + 1);
  }

  function reset() {
    setIsPlaying(false);
    setStep(0);
  }

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" onClick={prev} disabled={currentStep === 0}>
          <SkipBack className="w-4 h-4" />
        </Button>
        <Button size="sm" onClick={togglePlay} className="min-w-[80px]">
          {isPlaying ? <Pause className="w-4 h-4 mr-1" /> : <Play className="w-4 h-4 mr-1" />}
          {isPlaying ? labels.pause : labels.play}
        </Button>
        <Button data-testid="algorithm-player-next" variant="outline" size="sm" onClick={next} disabled={currentStep === maxStep}>
          <SkipForward className="w-4 h-4" />
        </Button>
        <Button variant="outline" size="sm" onClick={reset}>
          <RotateCcw className="w-4 h-4 mr-1" />
          {labels.reset}
        </Button>
        <div className="ml-auto flex items-center gap-2 text-sm text-gray-600">
          <span>速度</span>
          <div className="flex gap-1">
            {speeds.map((s, i) => (
              <button
                key={s}
                onClick={() => {
                  setSpeedIndex(i);
                  setIsPlaying(false);
                }}
                className={`px-2 py-0.5 rounded text-xs font-medium transition-colors ${
                  i === speedIndex
                    ? 'bg-blue-100 text-blue-700'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <span className="text-sm text-gray-600 min-w-[3rem]">{labels.step}</span>
        <Slider
          value={[currentStep]}
          min={0}
          max={maxStep}
          step={1}
          onValueChange={([v]) => {
            setIsPlaying(false);
            setStep(v);
          }}
          className="flex-1"
        />
        <span className="text-sm font-mono font-semibold min-w-[4rem] text-right">
          {currentStep} / {maxStep}
        </span>
      </div>
    </div>
  );
}
