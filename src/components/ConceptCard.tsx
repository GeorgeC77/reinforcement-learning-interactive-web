import type { ReactNode } from 'react';

interface ConceptCardProps {
  icon?: ReactNode;
  title: string;
  description: ReactNode;
  className?: string;
}

export default function ConceptCard({
  icon,
  title,
  description,
  className = '',
}: ConceptCardProps) {
  return (
    <div
      className={`bg-white border border-border-gray rounded-xl p-5 transition-all duration-200 hover:border-med-blue hover:-translate-y-0.5 hover:shadow-card ${className}`}
    >
      {icon && (
        <div className="w-10 h-10 rounded-lg bg-light-blue flex items-center justify-center text-med-blue mb-3">
          {icon}
        </div>
      )}
      <h4 className="text-lg font-semibold text-dark-gray mb-2">{title}</h4>
      <div className="text-[15px] leading-relaxed text-med-gray">
        {description}
      </div>
    </div>
  );
}
