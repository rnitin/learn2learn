#!/bin/bash 
#                                                    
##NECESSARY JOB SPECIFICATIONS                                  
#SBATCH --job-name=halfCheetahMAML3                                      
#SBATCH --time=8:00:00        
#SBATCH --gres=gpu:1                                    
#SBATCH --ntasks=8                                             
#SBATCH --ntasks-per-node=8                                    
#SBATCH --mem=10G                                             
#SBATCH --output=$SCRATCH/code/logs/halfCheetahMAML3.%j 

## Job commands
cd $SCRATCH/code/learn2learn/
export MUJOCO_PY_MUJOCO_PATH=$SCRATCH/sim/mujoco210
export LD_LIBRARY_PATH=$SCRATCH/sim/mujoco210/bin
python meta_rl/half_cheetah_maml.py --num_iterations=200 --adapt_steps=3
