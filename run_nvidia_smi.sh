#!/bin/bash
set -euo pipefail

echo "=== Hostname ==="
hostname
echo "=== Date ==="
date

IMG="/pnfs/knu.ac.kr/data/cms/store/user/yeo/singularity/swin2.sif"

cd /u/user/eoyun/4l/25.12.15/MergedElectronID 
echo "=== nvidia-smi inside container ==="
singularity exec --nv --bind /d0/scratch/eoyun:/d0/scratch/eoyun "$IMG" python3 Swin.py

echo "=== Done ==="

