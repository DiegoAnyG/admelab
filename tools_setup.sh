#!/usr/bin/env bash
# ============================================================================
# tools_setup.sh  ·  Portable JRE (Adoptium Temurin 21) + OPSIN into ./tools/
# For the self-verified IUPAC naming. No sudo required. Run:  bash tools_setup.sh
# The JRE is GPLv2 + Classpath Exception and is NOT committed to the repo.
# ============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
TOOLS="$HERE/tools"
mkdir -p "$TOOLS"
cd "$TOOLS"

# 1) Portable JRE 21
if [ -z "$(find . -maxdepth 1 -type d -name 'jdk-*' 2>/dev/null)" ]; then
  echo "Downloading Adoptium Temurin JRE 21..."
  curl -sL -o jre.tar.gz \
    "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jre/hotspot/normal/eclipse"
  tar xzf jre.tar.gz && rm -f jre.tar.gz
fi
JREDIR="$(find . -maxdepth 1 -type d -name 'jdk-*' | head -1)"
JAVA="$JREDIR/bin/java"
echo "JRE: $JREDIR"
"$JAVA" -version 2>&1 | head -2

# 2) OPSIN jar
if [ ! -s opsin.jar ]; then
  echo "Downloading OPSIN..."
  OPSIN_URL=$(curl -s https://api.github.com/repos/dan2097/opsin/releases/latest \
              | grep -o 'https://[^"]*jar-with-dependencies.jar' | head -1)
  curl -sL -o opsin.jar "$OPSIN_URL"
fi
echo "opsin.jar: $(du -h opsin.jar | cut -f1)"

echo "--- test name -> SMILES ---"
echo "2-acetyloxybenzoic acid" | "$JAVA" -jar opsin.jar -o smi
echo "OPSIN ready in $TOOLS"
