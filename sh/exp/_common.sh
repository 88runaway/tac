#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$ROOT_DIR" || exit 1
EVAL="bash policy/ACT/sh/eval.sh"
