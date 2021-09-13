#!/bin/sh
#SBATCH --job-name=test_QB
#SBATCH --mail-type=END
#SBATCH --mail-user=mastropi@uwalumni.com
#SBATCH --output=test_QB.slurm.out
date
python3.6 test_QB.py 3 J 15 9
date
