#!/bin/bash

FILES1=$(find configs/I20R5 -type f -name '*.yml')

for f in $FILES1
do
  python main.py $f

  current_time=$(date "+%Y.%m.%d-%H.%M.%S")
  git add .
  git commit -m "Finished $f at $current_time"
  git push origin main
done

FILES2=$(find configs/I5R20 -type f -name '*.yml')

for f in $FILES2
do
  python main.py $f

  current_time=$(date "+%Y.%m.%d-%H.%M.%S")
  git add .
  git commit -m "Finished $f at $current_time"
  git push origin main
done

FILES3=$(find configs/I5R5 -type f -name '*.yml')

for f in $FILES3
do
  python main.py $f

  current_time=$(date "+%Y.%m.%d-%H.%M.%S")
  git add .
  git commit -m "Finished $f at $current_time"
  git push origin main
done