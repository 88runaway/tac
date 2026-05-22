#!/bin/bash
# 一键终止所有 UniVTAC 评估相关进程
# Usage: bash sh/kill_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

PATTERNS=(
    'scripts/eval_policy.py'
    'scripts/parallel_eval_policy.py'
    'policy/ACT/sh/eval.sh'
    "${ROOT_DIR}/sh/exp/"
)

count_before=0
for pat in "${PATTERNS[@]}"; do
    n=$(pgrep -fc "$pat" 2>/dev/null || true)
    count_before=$((count_before + n))
done

if [ "$count_before" -eq 0 ]; then
    echo "未发现正在运行的评估进程。"
    exit 0
fi

echo "正在终止评估进程..."
for pat in "${PATTERNS[@]}"; do
    pkill -TERM -f "$pat" 2>/dev/null || true
done
sleep 2
for pat in "${PATTERNS[@]}"; do
    pkill -9 -f "$pat" 2>/dev/null || true
done

remaining=0
for pat in "${PATTERNS[@]}"; do
    n=$(pgrep -fc "$pat" 2>/dev/null || true)
    remaining=$((remaining + n))
done

if [ "$remaining" -eq 0 ]; then
    echo "已终止所有评估进程。"
else
    echo "警告: 仍有 ${remaining} 个进程未退出，请手动检查:"
    pgrep -af 'eval_policy|eval\.sh|UniVTAC/sh/exp' 2>/dev/null || true
    exit 1
fi
