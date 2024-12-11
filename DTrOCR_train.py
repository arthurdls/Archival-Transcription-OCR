import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import multiprocessing as mp
from typing import Tuple
import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from DTrOCR.dtrocr.processor import DTrOCRProcessor
from DTrOCR.dtrocr.config import DTrOCRConfig
from DTrOCR.dtrocr.model import DTrOCRLMHeadModel

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

# Data

print("Loading Data...")

def get_folder_names(path):
    folders = []
    for entry in os.scandir(path):
        if entry.is_dir():
            folders.append(entry.name)
    return folders

def get_data_dict(folder_names, data_directory):
    files = {name: [] for name in folder_names}
    for file_name, train_data_list in files.items():
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
                # 20% of data goes to validation
                if random.randint(1, 5) == 1:
                    validation_data_list.append(datapoint)
                else:
                    train_data_list.append(datapoint)
    return files

validation_data_list = []
train_data_list = []

directory_path = '../cursive_lines/'
folder_names = get_folder_names(directory_path)
cursive_lines_data = get_data_dict(folder_names, directory_path)
 

directory_path = '../scene_lines/'
folder_names = get_folder_names(directory_path)
scene_lines_data = get_data_dict(folder_names, directory_path)
 

directory_path = '../single_word/'
folder_names = get_folder_names(directory_path)
single_words_data = get_data_dict(folder_names, directory_path)

directory_path = '../scene_words/'
folder_names = get_folder_names(directory_path)
scene_words_data = get_data_dict(folder_names, directory_path)

# unify train data
for file_name, train_data_sublist in cursive_lines_data.items():
    train_data_list += train_data_sublist
for file_name, train_data_sublist in scene_lines_data.items():
    train_data_list += train_data_sublist
for file_name, train_data_sublist in single_words_data.items():
    train_data_list += train_data_sublist
for file_name, train_data_sublist in scene_words_data.items():
    train_data_list += train_data_sublist

print(f'Train size: {len(train_data_list)}; Validation size {len(validation_data_list)}')

# Data Loader

class SyntheticDataset(Dataset):
    def __init__(self, words, config: DTrOCRConfig):
        super(SyntheticDataset, self).__init__()
        self.words = words
        self.processor = DTrOCRProcessor(config, add_eos_token=True, add_bos_token=True)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        with Image.open(self.words[item]['image_path']).convert('RGB') as img:
            inputs = self.processor(
                images= img,
                texts=self.words[item]['text'],
                padding='max_length',
                return_tensors="pt",
                return_labels=True,
            )
        return {
            'pixel_values': inputs.pixel_values[0],
            'input_ids': inputs.input_ids[0],
            'attention_mask': inputs.attention_mask[0],
            'labels': inputs.labels[0]
        }

# download repos for offline use
config = DTrOCRConfig(
    # attn_implementation='flash_attention_2'
    # gpt2_hf_model = os.getcwd() + '/pretrained_repos/openai-community/gpt2',
    # vit_hf_model = os.getcwd() + '/pretrained_repos/google/vit-base-patch16-244',
    max_position_embeddings = 512
)

train_data = SyntheticDataset(words=train_data_list, config=config)
validation_data = SyntheticDataset(words=validation_data_list, config=config)

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=mp.cpu_count())
validation_dataloader = DataLoader(validation_data, batch_size=32, shuffle=False, num_workers=mp.cpu_count())

# Model

print("Loading Model...")

if torch.cuda.is_available():
    device = torch.device("cuda")  
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

torch.set_float32_matmul_precision('high')

model = DTrOCRLMHeadModel(config)
model = torch.compile(model)
model.to(device=device)

have_pretrained_model_location = False
if have_pretrained_model_location:
    START_EPOCH = 0
    if START_EPOCH:
        model.load_state_dict(torch.load(f'example_name_epoch_{START_EPOCH}_model_state_dict.pt', weights_only=True))
    else:
        model.load_state_dict(torch.load('example.pt', weights_only=True))

new_model_destination_folder = "trained_model/"
if not os.path.exists(new_model_destination_folder):
    os.makedirs(new_model_destination_folder)

print(model)

# Training

print("Training Model...")

def evaluate_model(model: torch.nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
    model.eval()

    losses, accuracies = [], []
    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader, total=len(dataloader), desc=f'Evaluating test set'):
            inputs = send_inputs_to_device(inputs, device=0)
            outputs = model(**inputs)

            losses.append(outputs.loss.item())
            accuracies.append(outputs.accuracy.item())

    loss = sum(losses) / len(losses)
    accuracy = sum(accuracies) / len(accuracies)
    model.train()
    return loss, accuracy

def send_inputs_to_device(dictionary, device):
    return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in dictionary.items()}

use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

EPOCHS = 30
train_losses, train_accuracies = [], []
validation_losses, validation_accuracies = [], []
for epoch in range(START_EPOCH, EPOCHS):
    epoch_losses, epoch_accuracies = [], []
    for inputs in tqdm.tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch {epoch + 1}'):

        # set gradients to zero
        optimizer.zero_grad()

        # send inputs to same device as model
        inputs = send_inputs_to_device(inputs, device=device)

        # forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            outputs = model(**inputs)

        # calculate gradients
        scaler.scale(outputs.loss).backward()

        # update weights
        scaler.step(optimizer)
        scaler.update()

        # update cumulative loss and accuracy
        epoch_losses.append(outputs.loss.item())
        epoch_accuracies.append(outputs.accuracy.item())

    # store loss and metrics
    train_losses.append(sum(epoch_losses) / len(epoch_losses))
    train_accuracies.append(sum(epoch_accuracies) / len(epoch_accuracies))

    # tests loss and accuracy
    validation_loss, validation_accuracy = evaluate_model(model, validation_dataloader)
    validation_losses.append(validation_loss)
    validation_accuracies.append(validation_accuracy)

    print(f"Epoch: {epoch + 1} - Train loss: {train_losses[-1]}, Train accuracy: {train_accuracies[-1]}, Validation loss: {validation_losses[-1]}, Validation accuracy: {validation_accuracies[-1]}")
    torch.save(model.state_dict(), f'{new_model_destination_folder}/epoch_{epoch + 1}_checkpoint_new_synthetic_model_state_dict.pt')

torch.save(model.state_dict(), f'{new_model_destination_folder}/trained_new_synthetic_model_state_dict.pt')