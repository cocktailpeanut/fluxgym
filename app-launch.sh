#!/usr/bin/env bash

cd "`dirname "$0"`" || exit 1
. env/bin/activate
python app.py
