#!/bin/bash

# disable buffering for stdout and stderr to avoid log output loss on app crash
export PYTHONUNBUFFERED=1

python3 app.py $@
