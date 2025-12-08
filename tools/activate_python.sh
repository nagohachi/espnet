#!/bin/bash
# Workaround for conda activate scripts expecting variables to be set
# Temporarily disable unbound variable errors during activation
set +u
eval "$(pixi shell-hook -s bash)"
set -u
