import { cn } from '@/lib/utils';

interface ValueHeatmapProps {
  rows: number;
  cols: number;
  values: number[];
  labels?: string[];
  min?: number;
  max?: number;
  positiveColor?: string;
  negativeColor?: string;
  zeroColor?: string;
  className?: string;
  onCellClick?: (index: number) => void;
}

export default function ValueHeatmap({
  rows,
  cols,
  values,
  labels,
  min,
  max,
  positiveColor = '#22c55e',
  negativeColor = '#ef4444',
  zeroColor = '#f3f4f6',
  className,
  onCellClick,
}: ValueHeatmapProps) {
  if (values.length !== rows * cols) {
    console.warn(`ValueHeatmap: expected ${rows * cols} values, got ${values.length}`);
  }
  const dataMin = min ?? Math.min(...values);
  const dataMax = max ?? Math.max(...values);
  const absMax = Math.max(Math.abs(dataMin), Math.abs(dataMax), 1e-6);

  function getColor(value: number) {
    if (value === 0) return zeroColor;
    const ratio = Math.abs(value) / absMax;
    const base = value > 0 ? positiveColor : negativeColor;
    return `color-mix(in srgb, ${base} ${ratio * 100}%, ${zeroColor})`;
  }

  return (
    <div
      className={cn('inline-grid gap-1', className)}
      style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
    >
      {values.map((value, idx) => (
        <button
          key={idx}
          onClick={() => onCellClick?.(idx)}
          disabled={!onCellClick}
          className={cn(
            'w-12 h-12 rounded-md flex items-center justify-center text-xs font-mono font-semibold transition-transform',
            onCellClick && 'hover:scale-105 cursor-pointer',
            !onCellClick && 'cursor-default'
          )}
          style={{
            backgroundColor: getColor(value),
            color: Math.abs(value) > absMax * 0.5 ? 'white' : '#1f2937',
          }}
          title={labels?.[idx] ?? `state ${idx}: ${value.toFixed(3)}`}
        >
          {value.toFixed(2)}
        </button>
      ))}
    </div>
  );
}
