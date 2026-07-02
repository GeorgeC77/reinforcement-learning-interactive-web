import type { ReactNode } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface InteractiveDemoProps {
  title: string;
  children: ReactNode;
}

export default function InteractiveDemo({ title, children }: InteractiveDemoProps) {
  return (
    <Card className="my-6 border-indigo-200 shadow-md">
      <CardHeader className="bg-indigo-50 border-b border-indigo-100">
        <CardTitle className="text-lg text-indigo-800">{title}</CardTitle>
      </CardHeader>
      <CardContent className="p-6">{children}</CardContent>
    </Card>
  );
}

export { InteractiveDemo };
