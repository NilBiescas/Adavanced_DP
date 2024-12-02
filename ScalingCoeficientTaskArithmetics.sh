#!/bin/bash
#SBATCH -n 2 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /fhome/amlai07/Adavanced_DP # working directory
#SBATCH -p tfg # Partition to submit to
#SBATCH --mem 2048 # 2GB solicitados.
#SBATCH -o error_folder/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e error_folder/%x_%u_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:1 # Para pedir gráficas

python3 /fhome/amlai07/Adavanced_DP/explore_task_arithmetics.py $1