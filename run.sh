#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=2
#PBS -l place=shared
#PBS -o coutput1.txt				
#PBS -e error1.txt				
#PBS -N frames				

cd ~/test												

source ~/.bashrc											
conda activate graftest	

module load cuda-12.4										
#python3 123.py	
#python3 eval.py configs/carla.yaml --pretrained --rotation_elevation
python train.py --config /Data/home/vicky/test/configs/RS307_0_i2.yaml