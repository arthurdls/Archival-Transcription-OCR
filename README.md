# Archival-Transcription-OCR

# Training and Testing DTrOCR

DTrOCR Original paper: https://doi.org/10.48550/arXiv.2308.15996 <br>
DTrOCR Model source: https://github.com/arvindrajan92/DTrOCR <br>
DTrOCR Training Data source: https://github.com/kris314/hwnet <br>

Code for training and testing the DTrOCR model is in the DTrOCR_code folder. Before running anything in the folder, make sure to read through the DTrOCR model source to ensure appropriate packages are installed. The code has various places where you can substitute file locations for personal use. You can also run the code offline by downloading the repositories corresponding to the base models for the DTrOCR as shown in the DTrOCR_code folder.

# Testing TrOCR

# Data
To acccess data used in project: https://drive.google.com/drive/folders/1gR1sH-_Lpj5Pz6p8S_GMBsQ7M8FOh98M?usp=sharing

Datasets <br>
IIIT-HWS: https://cvit.iiit.ac.in/research/projects/cvit-projects/matchdocimgs#iiit-hws <br>
Washington Historical Database: https://fki.tic.heia-fr.ch/databases/washington-database <br>


# Generating Synthetic Data
Synthetic data is generated using the pypi package trdg. For more information about TextRecognitionDataGenerator, see https://textrecognitiondatagenerator.readthedocs.io/en/latest/index.html


Before running the given script (generate_text.bat), download the "cursive fonts" folder for fonts and "sepia" folder for sepia backgrounds, from https://drive.google.com/drive/u/3/folders/1PRXvvmAHXTnRJ0rIBe5BzPtwFLXrItce

(note, current implementation of the trdg package runs in python 3.8 as of 12/10/2024) 

    conda create -n generate_env_3.8 python=3.8 
    conda activate generate_env_3.8     

To run on Windows Powershell: 
       
    pip install trdg 
    generate_text.bat 

