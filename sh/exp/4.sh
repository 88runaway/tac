#!/bin/bash
# GPU 3 - pull_out_key
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# $EVAL pull_out_key univtac     7
# $EVAL pull_out_key vision_only 7
$EVAL grasp_classify univtac     7
$EVAL grasp_classify vision_only 7
