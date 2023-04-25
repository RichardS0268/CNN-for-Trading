#!/bin/bash

FILES=$(find configs -type f -name '*.yml')

for f in $FILES
do
  python main.py $f

  current_time=$(date "+%Y.%m.%d-%H.%M.%S")
  git add .
  git commit -m "Finished $f at $current_time"
  git push origin main
done