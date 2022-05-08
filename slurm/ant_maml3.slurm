#!/bin/bash 
#                                                    
##NECESSARY JOB SPECIFICATIONS                                  
#SBATCH --job-name=antDirMAML3                                      
#SBATCH --time=8:00:00        
#SBATCH --gres=gpu:1                                    
#SBATCH --ntasks=8                                             
#SBATCH --ntasks-per-node=8                                    
#SBATCH --mem=10G                                             
#SBATCH --output=logs/antDirMAML3.%j 

## Job commands
cd $SCRATCH/code/learn2learn/
export MUJOCO_PY_MUJOCO_PATH=$SCRATCH/sim/mujoco210
export LD_LIBRARY_PATH=$SCRATCH/sim/mujoco210/bin
python meta_rl/antdir_maml.py --num_iterations=500 --adapt_steps=3
