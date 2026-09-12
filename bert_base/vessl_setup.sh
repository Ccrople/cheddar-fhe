#!/bin/bash
# Bring-up for the BERT-Base branch on the vessl A100 (the overlay is wiped
# whenever the laptop's session drops, so this is run often and is idempotent).
#   bash vessl_setup.sh [branch]   -> /root/work/cheddar-bb/build_bb
# Data is exported on the box by bb_export-style calls into /root/bert_base.
BRANCH="${1:-BERT_base}"
exec > >(tee -a /root/bb_setup.log) 2>&1
echo "==== $(date -u) setup start on $BRANCH"
export DEBIAN_FRONTEND=noninteractive
mkdir -p /root/work /root/bert_base

if [ -f /etc/apt/sources.list.d/cuda.list ] && \
   ! grep -q 'signed-by' /etc/apt/sources.list.d/cuda.list; then
  mv /etc/apt/sources.list.d/cuda.list /root/cuda.list.bak
fi
if [ ! -f /usr/include/x86_64-linux-gnu/gmp.h ]; then
  apt-get update -qq
  apt-get install -y -qq libgmp-dev build-essential ninja-build ca-certificates wget
fi
if [ ! -x /opt/cmake-3.31.6/bin/cmake ]; then
  mkdir -p /opt/cmake-3.31.6; cd /tmp
  wget -q https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz
  tar xzf cmake-3.31.6-linux-x86_64.tar.gz -C /opt/cmake-3.31.6 --strip-components=1
  rm -f cmake-3.31.6-linux-x86_64.tar.gz
fi
python3.12 -c "import numpy" 2>/dev/null || apt-get install -y -qq python3-numpy
python3.12 -c "import huggingface_hub, tokenizers" 2>/dev/null || {
  apt-get install -y -qq python3-pip
  python3.12 -m pip install --break-system-packages -q huggingface_hub tokenizers safetensors
}

if [ ! -d /root/work/cheddar-bb/.git ]; then
  git clone -q --branch "$BRANCH" https://github.com/Ccrople/cheddar-fhe.git /root/work/cheddar-bb || exit 1
fi
cd /root/work/cheddar-bb || exit 1
git fetch -q origin "$BRANCH" && git checkout -q -B "$BRANCH" "origin/$BRANCH"
echo "---- on $BRANCH at $(git rev-parse --short HEAD)"

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/opt/cmake-3.31.6/bin:$CUDA_HOME/bin:$PATH
echo "==== $(date -u) configure"
cmake -S . -B build_bb -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCMAKE_LIBRARY_PATH="$CUDA_HOME/lib64" \
  -DUSE_GMP=ON -DBUILD_UNITTEST=ON -DENABLE_EXTENSION=ON \
  -DUSE_CUBLAS=ON -DCPM_DOWNLOAD_ALL=ON > /root/bb_configure.log 2>&1 \
  || { echo "==== CONFIGURE FAILED"; tail -30 /root/bb_configure.log; exit 1; }
echo "==== $(date -u) building ci_bert_base_test"
cmake --build build_bb -j 48 --target ci_bert_base_test > /root/bb_compile.log 2>&1 \
  || { echo "==== BUILD FAILED"; grep -aE "error|Error" /root/bb_compile.log | head -30; exit 1; }
ls -l /root/work/cheddar-bb/build_bb/unittest/ci_bert_base_test
echo "==== $(date -u) BB_SETUP_DONE"
