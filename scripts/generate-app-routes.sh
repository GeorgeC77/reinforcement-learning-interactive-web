#!/usr/bin/env bash
set -euo pipefail

# Generate import statements and route entries for App.tsx
cd "$(dirname "$0")/.."

imports_file="scripts/app-imports.txt"
routes_file="scripts/app-routes.txt"
> "$imports_file"
> "$routes_file"

sections=(
  "chapter01/section-01-lms.tsx:Chapter01Section01Page"
  "chapter01/section-02.tsx:Chapter01Section02Page"
  "chapter01/section-03.tsx:Chapter01Section03Page"
  "chapter01/section-04.tsx:Chapter01Section04Page"
  "chapter02/section-01.tsx:Chapter02Section01Page"
  "chapter02/section-02.tsx:Chapter02Section02Page"
  "chapter02/section-03.tsx:Chapter02Section03Page"
  "chapter02/section-04.tsx:Chapter02Section04Page"
  "chapter03/section-01.tsx:Chapter03Section01Page"
  "chapter03/section-02.tsx:Chapter03Section02Page"
  "chapter04/section-01.tsx:Chapter04Section01Page"
  "chapter04/section-02.tsx:Chapter04Section02Page"
  "chapter05/section-01.tsx:Chapter05Section01Page"
  "chapter05/section-02.tsx:Chapter05Section02Page"
  "chapter05/section-03.tsx:Chapter05Section03Page"
  "chapter05/section-04.tsx:Chapter05Section04Page"
  "chapter06/section-01.tsx:Chapter06Section01Page"
  "chapter06/section-02.tsx:Chapter06Section02Page"
  "chapter07/section-01.tsx:Chapter07Section01Page"
  "chapter07/section-02.tsx:Chapter07Section02Page"
  "chapter07/section-03.tsx:Chapter07Section03Page"
  "chapter07/section-04.tsx:Chapter07Section04Page"
  "chapter07/section-05.tsx:Chapter07Section05Page"
  "chapter08/section-01.tsx:Chapter08Section01Page"
  "chapter08/section-02.tsx:Chapter08Section02Page"
  "chapter08/section-03.tsx:Chapter08Section03Page"
  "chapter09/section-01.tsx:Chapter09Section01Page"
  "chapter09/section-02.tsx:Chapter09Section02Page"
  "chapter09/section-03.tsx:Chapter09Section03Page"
  "chapter09/section-04.tsx:Chapter09Section04Page"
  "chapter10/section-01.tsx:Chapter10Section01Page"
  "chapter11/section-01.tsx:Chapter11Section01Page"
  "chapter11/section-02.tsx:Chapter11Section02Page"
  "chapter11/section-03.tsx:Chapter11Section03Page"
  "chapter11/section-04.tsx:Chapter11Section04Page"
  "chapter11/section-05.tsx:Chapter11Section05Page"
  "chapter12/section-01.tsx:Chapter12Section01Page"
  "chapter13/section-01.tsx:Chapter13Section01Page"
  "chapter14/section-01.tsx:Chapter14Section01Page"
  "chapter14/section-02.tsx:Chapter14Section02Page"
  "chapter14/section-03.tsx:Chapter14Section03Page"
  "chapter15/section-01.tsx:Chapter15Section01Page"
  "chapter15/section-02.tsx:Chapter15Section02Page"
  "chapter15/section-03.tsx:Chapter15Section03Page"
  "chapter15/section-04.tsx:Chapter15Section04Page"
  "chapter15/section-05.tsx:Chapter15Section05Page"
  "chapter16/section-01.tsx:Chapter16Section01Page"
  "chapter16/section-02.tsx:Chapter16Section02Page"
  "chapter16/section-03.tsx:Chapter16Section03Page"
  "chapter16/section-04.tsx:Chapter16Section04Page"
  "chapter17/section-01.tsx:Chapter17Section01Page"
)

for entry in "${sections[@]}"; do
  file="${entry%%:*}"
  component="${entry##*:}"
  path_slug="${file%.tsx}"
  echo "import ${component} from './pages/chapters/${path_slug}';" >> "$imports_file"

  # Extract chapter/section from path like chapter01/section-02
  chapter=$(echo "$path_slug" | sed -E 's|chapter([0-9]+)/section-([0-9]+)|\1|')
  section=$(echo "$path_slug" | sed -E 's|chapter([0-9]+)/section-([0-9]+)|\2|')
  route_path="/ch${chapter#0}/s${section#0}"
  echo "          <Route path=\"${route_path}\" element={<${component} />} />" >> "$routes_file"
done

echo "Generated imports in $imports_file and routes in $routes_file"
