#!/bin/bash
# add a parameter for the model name

#SBATCH --job-name=t_A_M
#SBATCH --nodes=1
#SBATCH --time=144:00:00
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/train_AF_mobilenet_pyramid_gate_unweight_8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
 
module load cuda/12.4
cd /ihome/stephenlee/anp407/Workspace/Adaptfilter
source /ihome/stephenlee/anp407/.bashrc
source activate adapt

python train_model_imagenet_20_new_gate_pyramid.py mobilenet 1 $1