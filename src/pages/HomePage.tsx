import { Link } from 'react-router-dom';
import { BookOpen, GraduationCap, ChevronRight, ShieldAlert, CheckCircle2, Construction } from 'lucide-react';
import { courseManifest, getChapterStatus, getCompletedCount, getTotalSectionCount, statusLabel, type Chapter, type SectionStatus } from '@/course/manifest';

function getChapterEntryPath(chapter: Chapter): string {
  return chapter.sections[0]?.path || '/';
}

function ChapterStatusIcon({ status }: { status: SectionStatus }) {
  switch (status) {
    case 'completed':
      return <CheckCircle2 className="w-4 h-4 text-emerald-600" />;
    case 'beta':
      return <div className="w-4 h-4 rounded-full bg-amber-500" />;
    case 'draft':
    default:
      return <Construction className="w-4 h-4 text-blue-600" />;
  }
}

export default function HomePage() {
  const chapters = courseManifest[0].chapters;
  const totalSections = getTotalSectionCount();
  const completedCount = getCompletedCount();

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-10">
      {/* Hero */}
      <section className="text-center py-14 bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center">
            <GraduationCap className="w-9 h-9 text-blue-600" />
          </div>
        </div>
        <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
          强化学习
        </h1>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto mb-6">
          一个交互式学习网站，从 MDP、Bellman 方程到策略梯度与 Actor-Critic，逐章深入理解强化学习的核心思想与算法。
        </p>

        {/* Copyright banner */}
        <p className="mt-6 text-sm text-amber-700 flex items-center justify-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          本内容仅供教学与非商业学习使用，完整授权说明见页脚。
        </p>

        {/* Progress stats */}
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3 text-sm">
          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
            <CheckCircle2 className="w-4 h-4" />
            已完成 {completedCount} / {totalSections}
          </div>
          {totalSections - completedCount > 0 && (
            <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200">
              <Construction className="w-4 h-4" />
              制作中 {totalSections - completedCount}
            </div>
          )}
        </div>
      </section>

      {/* Course Directory: 10 chapters flat */}
      <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
        <div className="flex items-center gap-3 mb-6">
          <BookOpen className="w-6 h-6 text-blue-600" />
          <h2 className="text-2xl font-bold text-gray-900">课程目录</h2>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {chapters.map((chapter) => {
            const entryPath = getChapterEntryPath(chapter);
            const chapterStatus = getChapterStatus(chapter);

            return (
              <Link
                key={chapter.id}
                to={entryPath}
                className="group flex flex-col p-5 rounded-xl border border-gray-200 bg-white hover:border-blue-300 hover:shadow-sm transition-all"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-grow min-w-0">
                    <div className="text-xs text-blue-600 font-semibold mb-1">
                      第 {chapter.number} 章
                    </div>
                    <div className="font-bold text-gray-900 group-hover:text-blue-600 transition-colors">
                      {chapter.title}
                    </div>
                    <div className="mt-2 inline-flex items-center gap-1 text-xs text-gray-500">
                      <ChapterStatusIcon status={chapterStatus} />
                      <span>{statusLabel(chapterStatus)}</span>
                    </div>
                  </div>
                  <ChevronRight className="w-4 h-4 text-gray-300 group-hover:text-blue-400 flex-shrink-0 mt-1" />
                </div>
                <div className="mt-3 text-sm text-gray-500 line-clamp-2">
                  {chapter.sections[0]?.description || ''}
                </div>
              </Link>
            );
          })}
        </div>
      </section>

      {/* License footer block */}
      <section className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-xl border border-amber-200 p-6">
        <div className="flex items-center gap-3 mb-3">
          <ShieldAlert className="w-6 h-6 text-amber-600" />
          <h3 className="text-lg font-bold text-amber-900">非商业用途</h3>
        </div>
        <p className="text-amber-800 text-sm leading-relaxed">
          本站所有原创内容版权归作者所有。你可以自由阅读、分享和用于个人学习，但禁止未经授权的商业使用。
          转载或引用请注明出处并遵守 CC BY-NC 4.0 许可协议。
        </p>
      </section>
    </div>
  );
}
