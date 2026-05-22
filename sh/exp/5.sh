#!/bin/bash
# GPU 4 - put_bottle_in_shelf
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
$EVAL put_bottle_in_shelf univtac     4
$EVAL put_bottle_in_shelf vision_only 4
