import { useEffect, useState, type Dispatch, type SetStateAction } from 'react';

const PREFIX = 'rl-demo:';

/**
 * useState backed by localStorage so demo experiment settings (seeds,
 * hyper-parameters) survive page reloads and can be shared by copying the URL.
 * Falls back to the initial value when storage is unavailable or corrupted.
 */
export function usePersistentState<T>(
  key: string,
  initial: T
): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(PREFIX + key);
      if (raw !== null) return JSON.parse(raw) as T;
    } catch {
      // Storage unavailable (private mode) or corrupted: use initial value.
    }
    return initial;
  });

  useEffect(() => {
    try {
      localStorage.setItem(PREFIX + key, JSON.stringify(value));
    } catch {
      // Ignore quota/security errors; the demo still works without persistence.
    }
  }, [key, value]);

  return [value, setValue];
}
