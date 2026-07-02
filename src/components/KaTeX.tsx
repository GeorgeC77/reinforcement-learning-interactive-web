import katex from 'katex';
import 'katex/dist/katex.min.css';
import { useMemo } from 'react';

interface KaTeXProps {
  math: string;
  display?: boolean;
  className?: string;
}

export default function KaTeX({ math, display = false, className = '' }: KaTeXProps) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(math, {
        displayMode: display,
        throwOnError: false,
        trust: true,
      });
    } catch {
      return math;
    }
  }, [math, display]);

  return (
    <span
      className={className}
      aria-label={math}
      title={math}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
