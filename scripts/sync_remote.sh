#!/bin/bash
# Push the repo (code, specs, lock) to the run hosts. Data and results never go this way.
cd "$(dirname "$0")/.."
for dest in "4090:<workstation>/dart-gxe/repo/" "amax:<server>/dart-gxe/repo/"; do
  rsync -a --exclude=/data --exclude=/results --exclude=/jobs --exclude=__pycache__ ./ "$dest" && echo "synced $dest"
done
