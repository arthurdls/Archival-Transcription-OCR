#!/bin/bash

# Loading Modules
source /etc/profile
#module load anaconda/2023a-tensorflow 

# change the file below to the file you would like to run 
wget http://ocr.iiit.ac.in/data/dataset/iiit-hws/iiit-hws.tar.gz
mkdir ground_truth/
cd ground_truth/
wget http://ocr.iiit.ac.in/data/dataset/iiit-hws/IIIT-HWS-90K.txt
wget http://ocr.iiit.ac.in/data/dataset/iiit-hws/IIIT-HWS-90K.txt
cd ..
tar -xvzf iiit-hws.tar.gz

# run LLsub submission_script.sh -s 12 -g volta:1
# use LLsub -h for help on what flags to use
