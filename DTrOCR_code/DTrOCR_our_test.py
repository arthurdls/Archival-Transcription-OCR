import os
import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from DTrOCR.dtrocr.processor import DTrOCRProcessor
from DTrOCR.dtrocr.config import DTrOCRConfig
from DTrOCR.dtrocr.model import DTrOCRLMHeadModel
from jiwer import cer

import warnings
warnings.filterwarnings("ignore")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.multiprocessing.set_sharing_strategy('file_system')

# Supercloud Environment Variables
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/usr/local/pkg/cuda/cuda-11.8'
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
os.environ['HF_HUB_OFFLINE']='1'

def get_folder_names(path):
    folders = []
    for entry in os.scandir(path):
        if entry.is_dir():
            folders.append(entry.name)
    return folders

def get_data_dict(folder_names, data_directory):
    files = {name: [] for name in folder_names}
    for file_name, test_data_list in files.items():
        with open(f'{data_directory}/{file_name}/labels.txt', 'r') as file:
            for line in file:
                split_line = line.replace("\n","").split(" ")
                img_path, target = split_line[0], " ".join(split_line[1:])
                if img_path == 'image':
                    continue
                datapoint = {
                    'image_path': os.getcwd() + f'/{data_directory}/{file_name}/' + img_path,
                    'text': target
                }
                test_data_list.append(datapoint)
    return files

directory_path = '../test_synth/'
folder_names = get_folder_names(directory_path)
files = get_data_dict(folder_names, directory_path)
print("Number of files: ", len(files))

# download repos for offline use
config = DTrOCRConfig(
    # attn_implementation='flash_attention_2'
    # gpt2_hf_model = os.getcwd() + '/pretrained_repos/openai-community/gpt2',
    # vit_hf_model = os.getcwd() + '/pretrained_repos/google/vit-base-patch16-244',
    max_position_embeddings = 512
)

# Model

print("Loading Model...")

torch.set_float32_matmul_precision('high')

model = DTrOCRLMHeadModel(config)
model = torch.compile(model)
model.load_state_dict(torch.load('./trained_model/retrain_old_singleword_model/epoch_16_checkpoint_new_synthetic_model_state_dict.pt', weights_only=True, map_location=torch.device(device)))
model.to('cpu')
model.eval()
print(model)

file_name = sys.argv[1]
print("Working on:", file_name)

test_data_list = files[file_name]
test_processor = DTrOCRProcessor(config)

i = 0
cer_scores = []
for datapoint in test_data_list:
    i += 1
    with Image.open(datapoint['image_path']).convert('RGB') as img:
        image = img
    actual_text = datapoint['text']

    inputs = test_processor(
        images=image.convert('RGB'),
        texts=test_processor.tokeniser.bos_token,
        return_tensors='pt'
    )

    model_output = model.generate(
        inputs,
        test_processor,
        num_beams=3
    )

    predicted_text = test_processor.tokeniser.decode(model_output[0], skip_special_tokens=True)
    cer_scores += [cer(actual_text, predicted_text)]

    # print(f"Actual: {actual_text}, Predicted: {predicted_text}, CER: {cer_scores[-1]} - {i}/{len(test_data_list)}")
    if i % 10 == 0:
        print(f"{i}/{len(test_data_list)}")
out = f"{file_name}: Average CER: {sum(cer_scores) / len(cer_scores)}"
print(out)

with open('result_cer.txt', 'a') as f:
    f.write(f"{out}\n")