import { NavLink, Link, useLocation } from 'react-router-dom';
import {
  GraduationCap,
  Home,
  ShieldAlert,
  ChevronDown,
  Menu,
  BookOpen,
  Grid3x3,
  Calculator,
  Trophy,
  RefreshCw,
  Dice5,
  TrendingDown,
  Clock,
  Brain,
  GitBranch,
  Activity,
  Route,
  Gift,
  BarChart3,
  MousePointer2,
  Scale,
  FlaskConical,
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import { cn } from '@/lib/utils';

type NavItem = { to: string; label: string; icon: React.ElementType };
type ChapterKey =
  | 'home'
  | 'ch01'
  | 'ch02'
  | 'ch03'
  | 'ch04'
  | 'ch05'
  | 'ch06'
  | 'ch07'
  | 'ch08'
  | 'ch09'
  | 'ch10';

const chapter01Items: NavItem[] = [
  { to: '/ch01/overview', label: '课程概览', icon: BookOpen },
  { to: '/ch01/mdp', label: '状态-动作-转移', icon: Grid3x3 },
  { to: '/ch01/policy', label: '策略与轨迹', icon: Route },
  { to: '/ch01/reward', label: '奖励设计', icon: Gift },
  { to: '/ch01/returns', label: '回报与马尔可夫性', icon: Clock },
];

const chapter02Items: NavItem[] = [
  { to: '/ch02/overview', label: '课程概览', icon: BookOpen },
  { to: '/ch02/bellman', label: '贝尔曼方程', icon: Calculator },
  { to: '/ch02/state-values', label: '状态值函数', icon: BarChart3 },
  { to: '/ch02/action-values', label: '动作值函数', icon: MousePointer2 },
];

const chapter03Items: NavItem[] = [
  { to: '/ch03/overview', label: '课程概览', icon: BookOpen },
  { to: '/ch03/boe', label: '贝尔曼最优方程', icon: Trophy },
];

const chapter04Items: NavItem[] = [
  { to: '/ch04/overview', label: '课程概览', icon: BookOpen },
  { to: '/ch04/vi-pi', label: '值迭代与策略迭代', icon: RefreshCw },
  { to: '/ch04/convergence', label: '收敛性与异步 DP', icon: TrendingDown },
];

const chapter05Items: NavItem[] = [
  { to: '/ch05/overview', label: '课程概览', icon: BookOpen },
  { to: '/ch05/mc', label: '蒙特卡洛方法', icon: Dice5 },
  { to: '/ch05/off-policy', label: '异策略 MC', icon: Scale },
];

const chapter06Items: NavItem[] = [
  { to: '/ch06/overview', label: '课程概览', icon: BookOpen },
  { to: '/ch06/sa', label: '随机逼近', icon: TrendingDown },
];

const chapter07Items: NavItem[] = [
  { to: '/ch07/overview', label: '课程概览', icon: BookOpen },
  { to: '/ch07/td', label: '时序差分', icon: Clock },
  { to: '/ch07/td-ext', label: '教材补充与拓展', icon: FlaskConical },
];

const chapter08Items: NavItem[] = [
  { to: '/ch08/overview', label: '课程概览', icon: BookOpen },
  { to: '/ch08/fa', label: '值函数近似', icon: Brain },
];

const chapter09Items: NavItem[] = [
  { to: '/ch09/overview', label: '课程概览', icon: BookOpen },
  { to: '/ch09/pg', label: '策略梯度', icon: GitBranch },
];

const chapter10Items: NavItem[] = [
  { to: '/ch10/overview', label: '课程概览', icon: BookOpen },
  { to: '/ch10/ac', label: 'Actor-Critic', icon: Activity },
];

const chapterConfig: Record<
  Exclude<ChapterKey, 'home'>,
  { label: string; icon: React.ElementType; items: NavItem[]; pathPrefix?: string }
> = {
  ch01: { label: '第 1 章 基本概念', icon: Grid3x3, items: chapter01Items, pathPrefix: '/ch01/' },
  ch02: { label: '第 2 章 贝尔曼方程', icon: Calculator, items: chapter02Items, pathPrefix: '/ch02/' },
  ch03: { label: '第 3 章 贝尔曼最优方程', icon: Trophy, items: chapter03Items, pathPrefix: '/ch03/' },
  ch04: { label: '第 4 章 值迭代与策略迭代', icon: RefreshCw, items: chapter04Items, pathPrefix: '/ch04/' },
  ch05: { label: '第 5 章 蒙特卡洛方法', icon: Dice5, items: chapter05Items, pathPrefix: '/ch05/' },
  ch06: { label: '第 6 章 随机逼近', icon: TrendingDown, items: chapter06Items, pathPrefix: '/ch06/' },
  ch07: { label: '第 7 章 时序差分方法', icon: Clock, items: chapter07Items, pathPrefix: '/ch07/' },
  ch08: { label: '第 8 章 值函数方法', icon: Brain, items: chapter08Items, pathPrefix: '/ch08/' },
  ch09: { label: '第 9 章 策略梯度方法', icon: GitBranch, items: chapter09Items, pathPrefix: '/ch09/' },
  ch10: { label: '第 10 章 Actor-Critic', icon: Activity, items: chapter10Items, pathPrefix: '/ch10/' },
};

function getCurrentChapter(path: string): ChapterKey {
  if (path === '/') return 'home';
  const entries = Object.entries(chapterConfig) as [
    Exclude<ChapterKey, 'home'>,
    { items: NavItem[]; pathPrefix?: string }
  ][];
  for (const [key, { items, pathPrefix }] of entries) {
    const paths = new Set(items.map((i) => i.to));
    if (paths.has(path) || (pathPrefix && path.startsWith(pathPrefix))) {
      return key;
    }
  }
  return 'ch01';
}

function ChapterNav({ chapter, currentPath }: { chapter: Exclude<ChapterKey, 'home'>; currentPath: string }) {
  const { label, icon: ChapterIcon, items } = chapterConfig[chapter];

  const renderItem = (item: NavItem) => {
    const isActive = currentPath === item.to;
    return (
      <Link
        key={item.to}
        to={item.to}
        className={cn(
          'flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors',
          isActive
            ? 'bg-blue-50 text-blue-700'
            : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
        )}
      >
        <item.icon className="w-4 h-4" />
        {item.label}
      </Link>
    );
  };

  return (
    <>
      {/* Desktop dropdown */}
      <div className="hidden lg:block">
        <DropdownMenu>
          <DropdownMenuTrigger className="flex items-center gap-1.5 px-3 py-2 rounded-md text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-50 transition-colors outline-none">
            <ChapterIcon className="w-4 h-4" />
            {label}
            <ChevronDown className="w-3.5 h-3.5" />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            {items.map((item) => {
              const isActive = currentPath === item.to;
              return (
                <DropdownMenuItem key={item.to} asChild>
                  <Link
                    to={item.to}
                    className={cn(
                      'flex items-center gap-2 cursor-pointer',
                      isActive && 'bg-blue-50 text-blue-700'
                    )}
                  >
                    <item.icon className="w-4 h-4" />
                    {item.label}
                  </Link>
                </DropdownMenuItem>
              );
            })}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Mobile sheet */}
      <div className="lg:hidden">
        <Sheet>
          <SheetTrigger asChild>
            <button
              type="button"
              className="flex items-center gap-1.5 px-3 py-2 rounded-md text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-50 transition-colors outline-none"
              aria-label="打开章节菜单"
            >
              <Menu className="w-4 h-4" />
              <span className="hidden sm:inline">章节</span>
            </button>
          </SheetTrigger>
          <SheetContent side="right" className="w-[280px] sm:w-80">
            <SheetHeader>
              <SheetTitle className="flex items-center gap-2 text-base">
                <ChapterIcon className="w-5 h-5 text-blue-600" />
                {label}
              </SheetTitle>
            </SheetHeader>
            <nav className="flex flex-col gap-1 py-4">{items.map(renderItem)}</nav>
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}

export default function Navbar() {
  const location = useLocation();
  const currentPath = location.pathname;
  const chapter = getCurrentChapter(currentPath);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/90 backdrop-blur-md border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          <NavLink
            to="/"
            className="flex items-center gap-2 text-lg font-bold text-blue-700 hover:text-blue-800 transition-colors"
          >
            <GraduationCap className="w-6 h-6" />
            <span className="hidden sm:inline">强化学习交互式课程</span>
            <span className="sm:hidden">RL 课程</span>
          </NavLink>

          <div className="flex items-center gap-1">
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-1.5 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-blue-50 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                )
              }
            >
              <Home className="w-4 h-4" />
              目录
            </NavLink>

            {chapter !== 'home' && <ChapterNav chapter={chapter} currentPath={currentPath} />}

            <a
              href="https://github.com/GeorgeC77/machine-learning-interactive-web/blob/main/LICENSE"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:flex items-center gap-1.5 px-3 py-2 rounded-md text-sm font-medium text-amber-700 bg-amber-50 hover:bg-amber-100 transition-colors"
              title="CC BY-NC 4.0 非商业许可"
            >
              <ShieldAlert className="w-4 h-4" />
              CC BY-NC 4.0
            </a>
          </div>
        </div>
      </div>
    </nav>
  );
}
