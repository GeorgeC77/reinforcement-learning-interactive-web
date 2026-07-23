import { RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

/**
 * Shared numeric seed input used by interactive demos so experiments are
 * reproducible. Changing the seed re-runs the same algorithm with the same
 * pseudo-random stream.
 */
export default function SeedControl({
  seed,
  onChange,
}: {
  seed: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex gap-2 items-end">
      <div className="flex-1">
        <label className="text-sm text-gray-600 mb-1 block">随机种子</label>
        <input
          type="number"
          value={seed}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full border rounded px-2 py-1 text-sm"
        />
      </div>
      <Button onClick={() => onChange(seed + 1)} variant="outline" size="sm">
        <RefreshCw className="w-4 h-4 mr-1" />
        换种子
      </Button>
    </div>
  );
}
