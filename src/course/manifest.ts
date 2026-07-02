export type SectionStatus = "draft" | "beta" | "completed";

export type Section = {
  id: string;
  title: string;
  path: string;
  status: SectionStatus;
  description?: string;
};

export type Chapter = {
  id: string;
  number: number;
  title: string;
  sections: Section[];
};

export type Part = {
  id: string;
  number: number;
  title: string;
  chapters: Chapter[];
};

export const courseManifest: Part[] = [
  {
    id: "part-rl",
    number: 1,
    title: "强化学习数学基础",
    chapters: [
      {
        id: "ch01",
        number: 1,
        title: "第 1 章 基本概念",
        sections: [
          {
            id: "ch01-overview",
            title: "1.0 课程概览",
            path: "/ch01/overview",
            status: "completed",
            description: "强化学习基本问题与学习路线。",
          },
          {
            id: "ch01-mdp",
            title: "1.1 马尔可夫决策过程",
            path: "/ch01/mdp",
            status: "completed",
            description: "状态、动作、奖励、转移与策略的形式化定义。",
          },
        ],
      },
      {
        id: "ch02",
        number: 2,
        title: "第 2 章 状态值与贝尔曼方程",
        sections: [
          {
            id: "ch02-overview",
            title: "2.0 课程概览",
            path: "/ch02/overview",
            status: "completed",
            description: "状态值函数与贝尔曼方程的学习路线。",
          },
          {
            id: "ch02-bellman",
            title: "2.1 贝尔曼方程",
            path: "/ch02/bellman",
            status: "completed",
            description: "贝尔曼方程的推导、矩阵形式与迭代求解。",
          },
        ],
      },
      {
        id: "ch03",
        number: 3,
        title: "第 3 章 最优状态值与贝尔曼最优方程",
        sections: [
          {
            id: "ch03-overview",
            title: "3.0 课程概览",
            path: "/ch03/overview",
            status: "completed",
            description: "最优值函数与贝尔曼最优方程。",
          },
          {
            id: "ch03-boe",
            title: "3.1 贝尔曼最优方程",
            path: "/ch03/boe",
            status: "completed",
            description: "最优策略、压缩映射与贪心策略。",
          },
        ],
      },
      {
        id: "ch04",
        number: 4,
        title: "第 4 章 值迭代与策略迭代",
        sections: [
          {
            id: "ch04-overview",
            title: "4.0 课程概览",
            path: "/ch04/overview",
            status: "completed",
            description: "值迭代与策略迭代的学习路线。",
          },
          {
            id: "ch04-vi-pi",
            title: "4.1 值迭代与策略迭代",
            path: "/ch04/vi-pi",
            status: "completed",
            description: "动态规划求解最优策略。",
          },
        ],
      },
      {
        id: "ch05",
        number: 5,
        title: "第 5 章 蒙特卡洛方法",
        sections: [
          {
            id: "ch05-overview",
            title: "5.0 课程概览",
            path: "/ch05/overview",
            status: "completed",
            description: "蒙特卡洛方法的学习路线。",
          },
          {
            id: "ch05-mc",
            title: "5.1 蒙特卡洛方法",
            path: "/ch05/mc",
            status: "completed",
            description: "MC Basic、Exploring Starts 与 ε-贪心。",
          },
        ],
      },
      {
        id: "ch06",
        number: 6,
        title: "第 6 章 随机逼近",
        sections: [
          {
            id: "ch06-overview",
            title: "6.0 课程概览",
            path: "/ch06/overview",
            status: "completed",
            description: "随机逼近与随机梯度下降。",
          },
          {
            id: "ch06-sa",
            title: "6.1 随机逼近",
            path: "/ch06/sa",
            status: "completed",
            description: "Robbins-Monro、Dvoretzky 与 SGD。",
          },
        ],
      },
      {
        id: "ch07",
        number: 7,
        title: "第 7 章 时序差分方法",
        sections: [
          {
            id: "ch07-overview",
            title: "7.0 课程概览",
            path: "/ch07/overview",
            status: "completed",
            description: "时序差分方法的学习路线。",
          },
          {
            id: "ch07-td",
            title: "7.1 时序差分学习",
            path: "/ch07/td",
            status: "completed",
            description: "TD(0)、Sarsa、n-步 Sarsa 与 Q-learning。",
          },
        ],
      },
      {
        id: "ch08",
        number: 8,
        title: "第 8 章 值函数方法",
        sections: [
          {
            id: "ch08-overview",
            title: "8.0 课程概览",
            path: "/ch08/overview",
            status: "completed",
            description: "值函数近似与深度 Q 学习。",
          },
          {
            id: "ch08-fa",
            title: "8.1 值函数近似",
            path: "/ch08/fa",
            status: "completed",
            description: "线性近似、LSTD 与 DQN。",
          },
        ],
      },
      {
        id: "ch09",
        number: 9,
        title: "第 9 章 策略梯度方法",
        sections: [
          {
            id: "ch09-overview",
            title: "9.0 课程概览",
            path: "/ch09/overview",
            status: "completed",
            description: "策略梯度方法的学习路线。",
          },
          {
            id: "ch09-pg",
            title: "9.1 策略梯度与 REINFORCE",
            path: "/ch09/pg",
            status: "completed",
            description: "策略梯度定理与 REINFORCE 算法。",
          },
        ],
      },
      {
        id: "ch10",
        number: 10,
        title: "第 10 章 Actor-Critic 方法",
        sections: [
          {
            id: "ch10-overview",
            title: "10.0 课程概览",
            path: "/ch10/overview",
            status: "completed",
            description: "Actor-Critic 方法的学习路线。",
          },
          {
            id: "ch10-ac",
            title: "10.1 Actor-Critic",
            path: "/ch10/ac",
            status: "completed",
            description: "QAC、A2C、离线 AC 与确定性 AC。",
          },
        ],
      },
    ],
  },
];

export function getAllSections(): Section[] {
  return courseManifest.flatMap((part) => part.chapters.flatMap((chapter) => chapter.sections));
}

export function getSectionByPath(path: string): Section | undefined {
  return getAllSections().find((section) => section.path === path);
}

export function getCompletedSections(): Section[] {
  return getAllSections().filter((section) => section.status === "completed");
}

export function getBetaSections(): Section[] {
  return getAllSections().filter((section) => section.status === "beta");
}

export function getDraftSections(): Section[] {
  return getAllSections().filter((section) => section.status === "draft");
}

export function getTotalSectionCount(): number {
  return getAllSections().length;
}

export function getCompletedCount(): number {
  return getCompletedSections().length;
}

export function getBetaCount(): number {
  return getBetaSections().length;
}

export function getDraftCount(): number {
  return getDraftSections().length;
}

export function getChapterStatus(chapter: Chapter): SectionStatus {
  const statuses = chapter.sections.map((s) => s.status);
  if (statuses.some((s) => s === "draft")) return "draft";
  if (statuses.some((s) => s === "beta")) return "beta";
  return "completed";
}

export function statusLabel(status: SectionStatus): string {
  switch (status) {
    case "draft":
      return "制作中";
    case "beta":
      return "预览版";
    case "completed":
      return "已完成";
    default:
      return "";
  }
}
