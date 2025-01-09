#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=1
#PBS -l place=shared
#PBS -o outputlabel.txt				
#PBS -e errorlabel.txt				
#PBS -N labelloc			

cd ~/test												

source ~/.bashrc											
conda activate graftest	

module load cuda-12.4										
#python3 123.py	
#python3 eval.py configs/carla.yaml --pretrained --rotation_elevation
python train.py --config /Data/home/vicky/test/configs/default.yaml