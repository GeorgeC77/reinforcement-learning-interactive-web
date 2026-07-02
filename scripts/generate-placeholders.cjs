const fs = require('fs');
const path = require('path');

// Simplified manifest data — must stay in sync with src/course/manifest.ts
const manifest = [
  {
    chapters: [
      { number: 1, sections: [
        { path: '/ch01/s02', componentName: 'Chapter01Section02Page' },
        { path: '/ch01/s03', componentName: 'Chapter01Section03Page' },
        { path: '/ch01/s04', componentName: 'Chapter01Section04Page' },
      ]},
      { number: 2, sections: [
        { path: '/ch02/s01', componentName: 'Chapter02Section01Page' },
        { path: '/ch02/s02', componentName: 'Chapter02Section02Page' },
        { path: '/ch02/s03', componentName: 'Chapter02Section03Page' },
        { path: '/ch02/s04', componentName: 'Chapter02Section04Page' },
      ]},
      { number: 3, sections: [
        { path: '/ch03/s01', componentName: 'Chapter03Section01Page' },
        { path: '/ch03/s02', componentName: 'Chapter03Section02Page' },
      ]},
      { number: 4, sections: [
        { path: '/ch04/s01', componentName: 'Chapter04Section01Page' },
        { path: '/ch04/s02', componentName: 'Chapter04Section02Page' },
      ]},
      { number: 5, sections: [
        { path: '/ch05/s01', componentName: 'Chapter05Section01Page' },
        { path: '/ch05/s02', componentName: 'Chapter05Section02Page' },
        { path: '/ch05/s03', componentName: 'Chapter05Section03Page' },
        { path: '/ch05/s04', componentName: 'Chapter05Section04Page' },
      ]},
      { number: 6, sections: [
        { path: '/ch06/s01', componentName: 'Chapter06Section01Page' },
        { path: '/ch06/s02', componentName: 'Chapter06Section02Page' },
      ]},
    ],
  },
  {
    chapters: [
      { number: 7, sections: [
        { path: '/ch07/s01', componentName: 'Chapter07Section01Page' },
        { path: '/ch07/s02', componentName: 'Chapter07Section02Page' },
        { path: '/ch07/s03', componentName: 'Chapter07Section03Page' },
        { path: '/ch07/s04', componentName: 'Chapter07Section04Page' },
        { path: '/ch07/s05', componentName: 'Chapter07Section05Page' },
      ]},
    ],
  },
  {
    chapters: [
      { number: 8, sections: [
        { path: '/ch08/s01', componentName: 'Chapter08Section01Page' },
        { path: '/ch08/s02', componentName: 'Chapter08Section02Page' },
        { path: '/ch08/s03', componentName: 'Chapter08Section03Page' },
      ]},
      { number: 9, sections: [
        { path: '/ch09/s01', componentName: 'Chapter09Section01Page' },
        { path: '/ch09/s02', componentName: 'Chapter09Section02Page' },
        { path: '/ch09/s03', componentName: 'Chapter09Section03Page' },
        { path: '/ch09/s04', componentName: 'Chapter09Section04Page' },
      ]},
    ],
  },
  {
    chapters: [
      { number: 10, sections: [
        { path: '/ch10/s01', componentName: 'Chapter10Section01Page' },
      ]},
      { number: 11, sections: [
        { path: '/ch11/s01', componentName: 'Chapter11Section01Page' },
        { path: '/ch11/s02', componentName: 'Chapter11Section02Page' },
        { path: '/ch11/s03', componentName: 'Chapter11Section03Page' },
        { path: '/ch11/s04', componentName: 'Chapter11Section04Page' },
        { path: '/ch11/s05', componentName: 'Chapter11Section05Page' },
      ]},
      { number: 12, sections: [
        { path: '/ch12/s01', componentName: 'Chapter12Section01Page' },
      ]},
      { number: 13, sections: [
        { path: '/ch13/s01', componentName: 'Chapter13Section01Page' },
      ]},
      { number: 14, sections: [
        { path: '/ch14/s01', componentName: 'Chapter14Section01Page' },
        { path: '/ch14/s02', componentName: 'Chapter14Section02Page' },
        { path: '/ch14/s03', componentName: 'Chapter14Section03Page' },
      ]},
    ],
  },
  {
    chapters: [
      { number: 15, sections: [
        { path: '/ch15/s01', componentName: 'Chapter15Section01Page' },
        { path: '/ch15/s02', componentName: 'Chapter15Section02Page' },
        { path: '/ch15/s03', componentName: 'Chapter15Section03Page' },
        { path: '/ch15/s04', componentName: 'Chapter15Section04Page' },
        { path: '/ch15/s05', componentName: 'Chapter15Section05Page' },
      ]},
      { number: 16, sections: [
        { path: '/ch16/s01', componentName: 'Chapter16Section01Page' },
        { path: '/ch16/s02', componentName: 'Chapter16Section02Page' },
        { path: '/ch16/s03', componentName: 'Chapter16Section03Page' },
        { path: '/ch16/s04', componentName: 'Chapter16Section04Page' },
      ]},
      { number: 17, sections: [
        { path: '/ch17/s01', componentName: 'Chapter17Section01Page' },
      ]},
    ],
  },
];

const template = (componentName, sectionPath) => `import SectionPlaceholder from '@/components/SectionPlaceholder';

export default function ${componentName}() {
  return <SectionPlaceholder sectionPath="${sectionPath}" />;
}
`;

function pathToFile(sectionPath) {
  const match = sectionPath.match(/^\/ch(\d+)\/s(\d+)$/);
  if (!match) throw new Error(`Invalid path: ${sectionPath}`);
  const chapter = match[1].padStart(2, '0');
  const section = match[2].padStart(2, '0');
  return path.join(__dirname, '..', 'src', 'pages', 'chapters', `chapter${chapter}`, `section-${section}.tsx`);
}

let created = 0;
for (const part of manifest) {
  for (const chapter of part.chapters) {
    for (const section of chapter.sections) {
      const filePath = pathToFile(section.path);
      fs.mkdirSync(path.dirname(filePath), { recursive: true });
      fs.writeFileSync(filePath, template(section.componentName, section.path));
      created++;
    }
  }
}

console.log(`Created ${created} placeholder section pages.`);
