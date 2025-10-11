#!/bin/bash
#SBATCH --job-name=qlora
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0958171@comp.nus.edu.sg
#SBATCH --gpus=h100-96

echo "==============================================================="
echo "Starting job on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "==============================================================="

echo "Activating Python virtual environment..."
source /home/e/e0958171/miniLM-LoRA/lora_env/bin/activate

PROJECT_DIR="/home/e/e0958171/miniLM-LoRA"
cd $PROJECT_DIR
echo "Current directory: $(pwd)"

echo "Starting LoRA training..."
python lora_main.py

echo "==============================================================="
echo "Job finished successfully!"
echo "==============================================================="


