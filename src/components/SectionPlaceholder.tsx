import { BookOpen, Construction, Home, ChevronLeft, ChevronRight, AlertTriangle, CheckCircle2, FlaskConical } from 'lucide-react';
import { Link } from 'react-router-dom';
import { getAllSections, getSectionByPath, statusLabel, type Section, type SectionStatus } from '@/course/manifest';

type SectionPlaceholderProps = {
  sectionPath: string;
};

function StatusIcon({ status }: { status: SectionStatus }) {
  switch (status) {
    case 'completed':
      return <CheckCircle2 className="w-4 h-4 text-emerald-600" />;
    case 'beta':
      return <FlaskConical className="w-4 h-4 text-amber-600" />;
    case 'draft':
    default:
      return <Construction className="w-4 h-4 text-blue-600" />;
  }
}

export default function SectionPlaceholder({ sectionPath }: SectionPlaceholderProps) {
  const section = getSectionByPath(sectionPath);
  if (!section) return null;

  const allSections = getAllSections();
  const currentIndex = allSections.findIndex((s) => s.path === sectionPath);
  const prevSection: Section | null = allSections[currentIndex - 1] ?? null;
  const nextSection: Section | null = allSections[currentIndex + 1] ?? null;

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-10">
      {/* Hero */}
      <section className="text-center py-12 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <BookOpen className="w-12 h-12 text-blue-600" />
        </div>
        <h1 className="text-4xl font-bold text-gray-900 mb-4">{section.title}</h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          {section.description}
        </p>

        {/* Copyright Notice */}
        <div className="mt-6 inline-flex items-center gap-2 bg-amber-50 border border-amber-300 rounded-lg px-5 py-3 max-w-3xl mx-auto">
          <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0" />
          <span className="text-sm font-medium text-amber-800">
            © 版权声明：本课程内容仅供个人学习交流使用，采用 CC BY-NC 4.0 许可。未经授权，严禁以任何形式用于商业用途，包括但不限于商业培训、付费课程、企业内训等。违者将依法追究法律责任。
          </span>
        </div>
      </section>

      {/* Coming Soon */}
      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
        <div className="flex flex-col items-center text-center space-y-6">
          <div className="w-20 h-20 bg-blue-50 rounded-full flex items-center justify-center">
            <Construction className="w-10 h-10 text-blue-600" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-3">内容建设中</h2>
            <p className="text-gray-600 max-w-xl mx-auto">
              这一小节的交互式教学内容正在精心制作中。你可以先浏览已经完成的章节，或者查看目录了解完整课程结构。
            </p>
          </div>
          <div className="flex flex-wrap justify-center gap-4">
            <Link
              to="/"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
            >
              <Home className="w-4 h-4" />
              返回课程目录
            </Link>
            {prevSection && (
              <Link
                to={prevSection.path}
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 rounded-lg font-medium transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                上一节
              </Link>
            )}
            {nextSection && (
              <Link
                to={nextSection.path}
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 rounded-lg font-medium transition-colors"
              >
                下一节
                <ChevronRight className="w-4 h-4" />
              </Link>
            )}
          </div>
        </div>
      </section>

      {/* Course info card */}
      <section className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200 p-6">
        <h3 className="text-lg font-bold text-blue-900 mb-3">关于本节</h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="bg-white/60 rounded-lg p-4">
            <span className="text-blue-700 font-semibold">小节标题：</span>
            <span className="text-gray-700">{section.title}</span>
          </div>
          <div className="bg-white/60 rounded-lg p-4">
            <span className="text-blue-700 font-semibold">完成状态：</span>
            <span className="text-gray-700 inline-flex items-center gap-1">
              <StatusIcon status={section.status} />
              {statusLabel(section.status)}
            </span>
          </div>
        </div>
      </section>
    </div>
  );
}
