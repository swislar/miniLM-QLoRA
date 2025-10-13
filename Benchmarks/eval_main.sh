#SBATCH --job-name=Evaluation
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0958171@comp.nus.edu.sg
#SBATCH --gpus=h100-96
#SBATCH --mem=64G

echo "==============================================================="
echo "Starting job on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "==============================================================="

echo "Activating Python virtual environment..."
source /home/e/e0958171/miniLM-LoRA/lora_env/bin/activate

PROJECT_DIR="/home/e/e0958171/miniLM-LoRA/Benchmarks"
cd $PROJECT_DIR
echo "Current directory: $(pwd)"

echo "Starting model evaluation..."
python eval.py

echo "==============================================================="
echo "Job finished successfully!"
echo "==============================================================="