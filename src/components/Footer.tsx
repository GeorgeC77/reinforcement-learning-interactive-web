import { ShieldAlert, ExternalLink, Mail } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="bg-gray-800 text-gray-300 py-8 mt-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-center md:text-left">
            <p className="font-semibold text-white">强化学习数学基础 · 交互式课程</p>
            <p className="text-sm text-gray-400 mt-1">
              基于 Shiyu Zhao《Mathematical Foundations of Reinforcement Learning》
            </p>
          </div>

          <a
            href="https://github.com/GeorgeC77/machine-learning-interactive-web/blob/main/LICENSE"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 bg-amber-500/10 border border-amber-500/30 rounded-lg text-amber-300 hover:bg-amber-500/20 hover:text-amber-200 transition-colors text-sm font-medium"
          >
            <ShieldAlert className="w-4 h-4" />
            <span>© CC BY-NC 4.0 · 非商业用途</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>

        <div className="mt-6 flex flex-col sm:flex-row items-center justify-center gap-2 text-sm text-gray-300">
          <Mail className="w-4 h-4 text-amber-300" />
          <span>如发现内容疏漏，或有合作意向，欢迎联系：</span>
          <a
            href="mailto:gengc25@hotmail.com"
            className="text-amber-300 hover:text-amber-200 underline font-medium"
          >
            gengc25@hotmail.com
          </a>
        </div>

        <div className="mt-6 pt-6 border-t border-gray-700 text-xs text-gray-400 text-center leading-relaxed">
          本课程内容仅供个人学习交流使用。未经授权，严禁以任何形式用于商业用途，包括但不限于商业培训、付费课程、企业内训等。
          转载或引用请注明出处并遵守{' '}
          <a
            href="https://creativecommons.org/licenses/by-nc/4.0/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-amber-300 hover:text-amber-200 underline"
          >
            CC BY-NC 4.0
          </a>{' '}
          许可协议。违者将依法追究法律责任。
        </div>
      </div>
    </footer>
  );
}
