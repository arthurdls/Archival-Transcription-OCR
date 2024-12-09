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

from DTrOCR.dtrocr.processor import DTrOCRProcessor, modified_build_inputs_with_special_tokens
from DTrOCR.dtrocr.config import DTrOCRConfig
from DTrOCR.dtrocr.model import DTrOCRLMHeadModel

import warnings
warnings.filterwarnings("ignore")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/usr/local/pkg/cuda/cuda-11.8'
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
os.environ['HF_HUB_OFFLINE']='1'
torch.multiprocessing.set_sharing_strategy('file_system')

print("Loading Data...")

files = [
    'white_line',
    'white_blur_line',
    'white_distort_line',
    'noise_line',
    'noise_blur_line',
    'noise_distort_line',
    'sepia_line',
    'sepia_blur_line',
    'sepia_distort_line'
]

cursive_lines_data = {file:[] for file in files}
scene_lines_data = {file:[] for file in files}
validation_data_list = []
train_data_list = []

single_words_data = {
    'noise_word':[],
    'noise_word_blur':[],
    'noise_word_distort':[],
    'sepia_word':[],
    'sepia_word_blur':[],
    'sepia_word_distort':[],
    'white_word':[],
    'white_word_blur':[],
    'white_word_distort':[]
}
scene_words_data = {
    'noise_word':[],
    'noise_blur_word':[],
    'noise_distort_word':[],
    'sepia_word':[],
    'sepia_blur_word':[],
    'sepia_distort_word':[],
    'white_word':[],
    'white_blur_word':[],
    'white_distort_word':[]
}


for file_name, train_data_sublist in scene_words_data.items():
    with open(f'../scene_words/{file_name}/labels.txt', 'r') as file:
        for line in file:
            img_path, target = line.replace(",", "").replace("\n","").split(" ")
            if img_path == 'image':
                continue
            datapoint = {
                'image_path': os.getcwd() + f'/../scene_words/{file_name}/' + img_path,
                'text': target
            }
            # 20% of data goes to validation
            if random.randint(1, 5) == 1:
                validation_data_list.append(datapoint)
            else:
                train_data_sublist.append(datapoint)

for file_name, train_data_sublist in single_words_data.items():
    with open(f'../single_word/{file_name}/labels.txt', 'r') as file:
        for line in file:
            img_path, target = line.replace(",", "").replace("\n","").split(" ")
            if img_path == 'image':
                continue
            datapoint = {
                'image_path': os.getcwd() + f'/../single_word/{file_name}/' + img_path,
                'text': target
            }
            # 20% of data goes to validation
            if random.randint(1, 5) == 1:
                validation_data_list.append(datapoint)
            else:
                train_data_sublist.append(datapoint)

for file_name, train_data_sublist in cursive_lines_data.items():
    with open(f'../cursive_lines/{file_name}/labels.txt', 'r') as file:
        for line in file:
            split_line = line.replace(",", "").replace("\n","").split(" ")
            img_path, target = split_line[0], " ".join(split_line[1:])
            if img_path == 'image':
                continue
            datapoint = {
                'image_path': os.getcwd() + f'/../cursive_lines/{file_name}/' + img_path,
                'text': target
            }
            # 20% of data goes to validation
            if random.randint(1, 5) == 1:
                validation_data_list.append(datapoint)
            else:
                train_data_sublist.append(datapoint)

for file_name, train_data_sublist in scene_lines_data.items():
    with open(f'../scene_lines/{file_name}/labels.txt', 'r') as file:
        for line in file:
            split_line = line.replace(",", "").replace("\n","").split(" ")
            img_path, target = split_line[0], " ".join(split_line[1:])

            if img_path == 'image':
                continue
            datapoint = {
                'image_path': os.getcwd() + f'/../scene_lines/{file_name}/' + img_path,
                'text': target
            }
            # 20% of data goes to validation
            if random.randint(1, 5) == 1:
                validation_data_list.append(datapoint)
            else:
                train_data_sublist.append(datapoint)

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

config = DTrOCRConfig(
    # attn_implementation='flash_attention_2'
    gpt2_hf_model = os.getcwd() + '/pretrained_repos/openai-community/gpt2',
    vit_hf_model = os.getcwd() + '/pretrained_repos/google/vit-base-patch16-244',
    max_position_embeddings = 256
)

print("THIS MODEL HAS 256 MAX POSITIONAL EMBEDDING")

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

START_EPOCH = 0
new_model_destination_folder = "trained_model/equal_scene_cursive_data_256_max_positional"
if not os.path.exists(new_model_destination_folder):
    os.makedirs(new_model_destination_folder)

model = DTrOCRLMHeadModel(config)
model = torch.compile(model)
model.to(device=device)
# model.load_state_dict(torch.load(f'./trained_model/epoch_{START_EPOCH}_checkpoint_new_synthetic_model_state_dict.pt', weights_only=True))
print(model)

# Training

print("Training Model...")

def evaluate_model(model: torch.nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
    # set model to evaluation mode
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

    # set model back to training mode
    model.train()

    return loss, accuracy

def send_inputs_to_device(dictionary, device):
    return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in dictionary.items()}

use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

EPOCHS = 80
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

        epoch_losses.append(outputs.loss.item())
        epoch_accuracies.append(outputs.accuracy.item())

        # iters = len(epoch_losses)
        # if iters % 100 == 0:
        #     print(f"Iter: {iters} - Loss: {sum(epoch_losses[-100:])/100} - Accuracy: {sum(epoch_accuracies[-100:])/100}")

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