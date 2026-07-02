#!/usr/bin/env bash
set -euo pipefail

# List of sections to generate: "chapter section componentName"
sections=(
  "01 02 Chapter01Section02Page"
  "01 03 Chapter01Section03Page"
  "01 04 Chapter01Section04Page"
  "02 01 Chapter02Section01Page"
  "02 02 Chapter02Section02Page"
  "02 03 Chapter02Section03Page"
  "02 04 Chapter02Section04Page"
  "03 01 Chapter03Section01Page"
  "03 02 Chapter03Section02Page"
  "04 01 Chapter04Section01Page"
  "04 02 Chapter04Section02Page"
  "05 01 Chapter05Section01Page"
  "05 02 Chapter05Section02Page"
  "05 03 Chapter05Section03Page"
  "05 04 Chapter05Section04Page"
  "06 01 Chapter06Section01Page"
  "06 02 Chapter06Section02Page"
  "07 01 Chapter07Section01Page"
  "07 02 Chapter07Section02Page"
  "07 03 Chapter07Section03Page"
  "07 04 Chapter07Section04Page"
  "07 05 Chapter07Section05Page"
  "08 01 Chapter08Section01Page"
  "08 02 Chapter08Section02Page"
  "08 03 Chapter08Section03Page"
  "09 01 Chapter09Section01Page"
  "09 02 Chapter09Section02Page"
  "09 03 Chapter09Section03Page"
  "09 04 Chapter09Section04Page"
  "10 01 Chapter10Section01Page"
  "11 01 Chapter11Section01Page"
  "11 02 Chapter11Section02Page"
  "11 03 Chapter11Section03Page"
  "11 04 Chapter11Section04Page"
  "11 05 Chapter11Section05Page"
  "12 01 Chapter12Section01Page"
  "13 01 Chapter13Section01Page"
  "14 01 Chapter14Section01Page"
  "14 02 Chapter14Section02Page"
  "14 03 Chapter14Section03Page"
  "15 01 Chapter15Section01Page"
  "15 02 Chapter15Section02Page"
  "15 03 Chapter15Section03Page"
  "15 04 Chapter15Section04Page"
  "15 05 Chapter15Section05Page"
  "16 01 Chapter16Section01Page"
  "16 02 Chapter16Section02Page"
  "16 03 Chapter16Section03Page"
  "16 04 Chapter16Section04Page"
  "17 01 Chapter17Section01Page"
)

count=0
for entry in "${sections[@]}"; do
  read -r chapter section componentName <<< "$entry"
  dir="src/pages/chapters/chapter${chapter}"
  file="${dir}/section-${section}.tsx"
  path="/ch${chapter}/s${section}"
  mkdir -p "$dir"
  cat > "$file" <<EOF
import SectionPlaceholder from '@/components/SectionPlaceholder';

export default function ${componentName}() {
  return <SectionPlaceholder sectionPath="${path}" />;
}
EOF
  count=$((count + 1))
done

echo "Created $count placeholder section pages."
