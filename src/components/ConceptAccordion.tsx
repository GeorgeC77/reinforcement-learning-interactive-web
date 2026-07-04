import type { ReactNode } from 'react';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';

interface ConceptItem {
  id: string;
  title: string;
  content: ReactNode;
}

interface ConceptAccordionProps {
  items: ConceptItem[];
  defaultOpen?: string[];
  className?: string;
}

export default function ConceptAccordion({
  items,
  defaultOpen,
  className = '',
}: ConceptAccordionProps) {
  return (
    <Accordion type="multiple" defaultValue={defaultOpen} className={className}>
      {items.map((item) => (
        <AccordionItem key={item.id} value={item.id} className="border-b border-gray-200">
          <AccordionTrigger className="text-left text-base font-medium text-gray-800 hover:text-blue-700 py-3">
            {item.title}
          </AccordionTrigger>
          <AccordionContent className="text-sm text-gray-600 leading-relaxed pb-4">
            {item.content}
          </AccordionContent>
        </AccordionItem>
      ))}
    </Accordion>
  );
}
