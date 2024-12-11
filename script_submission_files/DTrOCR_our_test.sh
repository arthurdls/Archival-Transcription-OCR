#!/bin/bash

# Loading Modules
source /etc/profile
module load anaconda/2023a-pytorch 

# change the file below to the file you would like to run 
python -u DTrOCR_our_test.py hw_sepia_distort_line

# run LLsub submission_script.sh -s 12 -g volta:1
# use LLsub -h for help on what flags to use
