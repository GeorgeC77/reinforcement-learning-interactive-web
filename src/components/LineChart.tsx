import type { ReactNode } from 'react';
import {
  LineChart as ReLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface Series {
  key: string;
  name: string;
  color?: string;
  strokeDasharray?: string;
}

interface LineChartProps {
  data: Record<string, number | string>[];
  xKey: string;
  xLabel?: string;
  yLabel?: string;
  series: Series[];
  height?: number;
  logY?: boolean;
  children?: ReactNode;
}

export default function LineChart({
  data,
  xKey,
  xLabel,
  yLabel,
  series,
  height = 240,
  logY = false,
  children,
}: LineChartProps) {
  const colors = ['#2563eb', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899'];

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <ReLineChart data={data} margin={{ top: 8, right: 16, bottom: 24, left: 24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey={xKey}
            tick={{ fontSize: 12, fill: '#6b7280' }}
            label={{ value: xLabel, position: 'insideBottom', offset: -12, fontSize: 12, fill: '#6b7280' }}
          />
          <YAxis
            scale={logY ? 'log' : 'auto'}
            domain={logY ? ['auto', 'auto'] : undefined}
            tickFormatter={logY ? (v: number) => v.toExponential(0) : undefined}
            tick={{ fontSize: 12, fill: '#6b7280' }}
            label={{ value: yLabel, angle: -90, position: 'insideLeft', fontSize: 12, fill: '#6b7280' }}
          />
          <Tooltip
            contentStyle={{ borderRadius: 8, border: '1px solid #e5e7eb' }}
            labelStyle={{ color: '#374151' }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {series.map((s, i) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.name}
              stroke={s.color ?? colors[i % colors.length]}
              strokeWidth={2}
              strokeDasharray={s.strokeDasharray}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </ReLineChart>
      </ResponsiveContainer>
      {children && <div className="mt-2">{children}</div>}
    </div>
  );
}
