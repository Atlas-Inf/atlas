#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix: which external endpoints the runner can reach (for engine-comparison jobs that need sources and models).
set -u
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
{
  echo "proxy env: $(env | grep -iE '^(https?|no|all)_proxy=' | sed 's/=.*/=set/' | tr '\n' ' ')"
  echo "HF_ENDPOINT=${HF_ENDPOINT:-unset}"
  git config --global --get-regexp 'url\..*insteadof' 2>/dev/null | sed 's/^/git insteadOf: /'
  for u in https://github.com https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/heads/master https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
           https://objects.githubusercontent.com https://pypi.org/simple/gguf/ https://files.pythonhosted.org https://download.pytorch.org/whl/cpu/ \
           https://gufo.cachix.org/nix-cache-info https://cache.nixos.org/nix-cache-info https://ghcr.io/v2/ https://registry-1.docker.io/v2/ \
           https://huggingface.co/api/models/unsloth/Qwen3.8-Flash-Next-GGUF "${HF_ENDPOINT:-https://huggingface.co}/api/models/unsloth/Qwen3.8-Flash-Next-GGUF"; do
    c=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$u"); echo "  $c  $u"
  done
  timeout 20 git ls-remote --heads https://github.com/ggml-org/llama.cpp.git master 2>&1 | head -1 | sed 's/^/git ls-remote: /'
  docker info 2>/dev/null | grep -iE "registry mirrors|proxy" -A2 | head -6
  pip config list 2>/dev/null | head -3; cat /etc/pip.conf 2>/dev/null | head -5
} 2>&1 | tee -a "$RES"
