import { useMemo } from 'react';
import {
  stateToRowCol,
} from '@/lib/rl/gridworld';
import type {
  GridWorldConfig,
  Policy,
  StateValues,
} from '@/lib/rl/gridworld';

export interface GridWorldProps {
  config: GridWorldConfig;
  policy?: Policy;
  values?: StateValues;
  trajectory?: number[];
  currentStep?: number;
  highlightState?: number | null;
  highlightNextState?: number | null;
  highlightUpdatedState?: number | null;
  highlightAction?: { state: number; action: number } | null;
  onCellClick?: (state: number) => void;
  onActionClick?: (state: number, action: number) => void;
  editable?: boolean;
  showValues?: boolean;
  className?: string;
}

const CELL_SIZE = 72;
const GAP = 8;

export default function GridWorld({
  config,
  policy,
  values,
  trajectory = [],
  currentStep = -1,
  highlightState = null,
  highlightNextState = null,
  highlightUpdatedState = null,
  highlightAction = null,
  onCellClick,
  onActionClick,
  editable = false,
  showValues = false,
  className = '',
}: GridWorldProps) {
  const numStates = config.rows * config.cols;

  const valueRange = useMemo(() => {
    if (!values || values.length === 0) return { min: -1, max: 1 };
    const min = Math.min(...values);
    const max = Math.max(...values);
    if (min === max) return { min: min - 1, max: max + 1 };
    return { min, max };
  }, [values]);

  function valueColor(value: number): string {
    const t = (value - valueRange.min) / (valueRange.max - valueRange.min);
    // Interpolate between soft red (low) and accent teal (high)
    const r = Math.round(226 + (0 - 226) * t);
    const g = Math.round(91 + (180 - 91) * t);
    const b = Math.round(91 + (166 - 91) * t);
    return `rgb(${r}, ${g}, ${b})`;
  }

  function cellFill(state: number): string {
    if (state === config.targetState) return '#e8f5e9'; // light green
    if (config.forbiddenStates.includes(state)) return '#ffebee'; // light red
    if (state === config.startState) return '#e3f2fd'; // light blue
    return '#ffffff';
  }

  function cellStroke(state: number): string {
    if (state === highlightState) return '#3a7bd5';
    if (state === highlightNextState) return '#f59e0b';
    if (state === config.targetState) return '#4caf50';
    if (config.forbiddenStates.includes(state)) return '#e25b5b';
    if (state === config.startState) return '#3a7bd5';
    if (state === highlightUpdatedState) return '#22c55e';
    return '#dfe6e9';
  }

  const width = config.cols * CELL_SIZE + (config.cols - 1) * GAP;
  const height = config.rows * CELL_SIZE + (config.rows - 1) * GAP;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={`select-none ${className}`}
    >
      {Array.from({ length: numStates }, (_, state) => {
        const { row, col } = stateToRowCol(state, config.cols);
        const x = col * (CELL_SIZE + GAP);
        const y = row * (CELL_SIZE + GAP);
        const isHighlighted = state === highlightState;
        const isCurrent = trajectory[currentStep] === state;
        const isInTrajectory = trajectory.includes(state);

        return (
          <g key={state}>
            {/* Background with optional value heatmap */}
            <rect
              x={x}
              y={y}
              width={CELL_SIZE}
              height={CELL_SIZE}
              rx={10}
              fill={showValues && values ? valueColor(values[state]) : cellFill(state)}
              stroke={cellStroke(state)}
              strokeWidth={isHighlighted ? 3 : 2}
              className={editable ? 'cursor-pointer' : ''}
              onClick={() => onCellClick?.(state)}
            />

            {/* State label */}
            <text
              x={x + 10}
              y={y + 18}
              className="text-xs font-bold fill-gray-500"
              style={{ fontSize: 12 }}
            >
              s{state + 1}
            </text>

            {/* Role label */}
            <text
              x={x + CELL_SIZE / 2}
              y={y + CELL_SIZE - 10}
              textAnchor="middle"
              className="text-[10px] fill-gray-400"
              style={{ fontSize: 10 }}
            >
              {state === config.startState && '起点'}
              {state === config.targetState && '目标'}
              {config.forbiddenStates.includes(state) && '禁区'}
            </text>

            {/* Value display */}
            {showValues && values && (
              <text
                x={x + CELL_SIZE / 2}
                y={y + CELL_SIZE / 2 - 10}
                textAnchor="middle"
                className="text-sm font-semibold fill-gray-800"
                style={{ fontSize: 13 }}
              >
                {values[state].toFixed(2)}
              </text>
            )}

            {/* Policy arrows: render all 5 actions so every direction is clickable.
                Stay (action 4) is rendered first so it does not block directional arrows. */}
            {policy && (
              <>
                <PolicyArrow
                  x={x}
                  y={y}
                  action={4}
                  prob={policy[state][4]}
                  editable={editable}
                  highlighted={
                    highlightAction?.state === state && highlightAction?.action === 4
                  }
                  onClick={() => onActionClick?.(state, 4)}
                />
                {[0, 1, 2, 3].map((action) => (
                  <PolicyArrow
                    key={action}
                    x={x}
                    y={y}
                    action={action}
                    prob={policy[state][action]}
                    editable={editable}
                    highlighted={
                      highlightAction?.state === state && highlightAction?.action === action
                    }
                    onClick={() => onActionClick?.(state, action)}
                  />
                ))}
              </>
            )}

            {/* Trajectory dot */}
            {isInTrajectory && (
              <circle
                cx={x + CELL_SIZE - 12}
                cy={y + 12}
                r={4}
                fill="#3a7bd5"
                opacity={isCurrent ? 1 : 0.4}
              />
            )}

            {/* Current step marker */}
            {isCurrent && (
              <circle
                cx={x + CELL_SIZE / 2}
                cy={y + CELL_SIZE / 2}
                r={14}
                fill="none"
                stroke="#1a3a5c"
                strokeWidth={3}
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}

function PolicyArrow({
  x,
  y,
  action,
  prob,
  editable,
  highlighted,
  onClick,
}: {
  x: number;
  y: number;
  action: number;
  prob: number;
  editable: boolean;
  highlighted?: boolean;
  onClick: () => void;
}) {
  const cx = x + CELL_SIZE / 2;
  const cy = y + CELL_SIZE / 2 + 4;
  const isSelected = prob > 0 || highlighted;

  if (action === 4) {
    // stay indicator: small square at bottom-right, does not block arrows
    return (
      <g
        className={editable ? 'cursor-pointer' : ''}
        onClick={onClick}
      >
        <rect
          x={x + CELL_SIZE - 16}
          y={y + CELL_SIZE - 16}
          width={8}
          height={8}
          rx={2}
          fill={isSelected ? '#1a3a5c' : 'transparent'}
          stroke={isSelected ? '#1a3a5c' : '#b0b8c0'}
          strokeWidth={2}
          opacity={isSelected ? 1 : 0.5}
        />
      </g>
    );
  }

  const len = isSelected ? 18 : 14;
  const angle = [-90, 0, 90, 180][action] * (Math.PI / 180);
  const x2 = cx + len * Math.cos(angle);
  const y2 = cy + len * Math.sin(angle);
  const color = highlighted ? '#ef4444' : isSelected ? '#1a3a5c' : '#b0b8c0';
  const opacity = highlighted ? 1 : isSelected ? 1 : 0.45;
  const markerId = `arrowhead-${action}-${isSelected ? 1 : 0}-${x}-${y}`;

  return (
    <g
      className={editable ? 'cursor-pointer' : ''}
      onClick={onClick}
    >
      {/* Invisible wide click target */}
      <line
        x1={cx}
        y1={cy}
        x2={x2}
        y2={y2}
        stroke="transparent"
        strokeWidth={18}
      />
      {/* Visible arrow */}
      <line
        x1={cx}
        y1={cy}
        x2={x2}
        y2={y2}
        stroke={color}
        strokeWidth={isSelected ? 3 : 2}
        opacity={opacity}
        markerEnd={`url(#${markerId})`}
      />
      <defs>
        <marker
          id={markerId}
          markerWidth={8}
          markerHeight={6}
          refX={7}
          refY={3}
          orient="auto"
        >
          <polygon points="0 0, 8 3, 0 6" fill={color} opacity={opacity} />
        </marker>
      </defs>
    </g>
  );
}
