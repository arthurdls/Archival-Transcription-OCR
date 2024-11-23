#!/bin/bash

# Loading Modules
source /etc/profile
module load anaconda/2023a-pytorch 

# change the file below to the file you would like to run 
# pip install -r DTrOCR/requirements.txt
python DTrOCR_train.py

# run LLsub submission_script.sh -s 12 -g volta:1
# use LLsub -h for help on what flags to use
