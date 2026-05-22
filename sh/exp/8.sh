#!/bin/bash
# GPU 7 - grasp_classify + insert_card
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
$EVAL grasp_classify univtac     7
$EVAL grasp_classify vision_only 7

