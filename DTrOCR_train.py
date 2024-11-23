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

print("Loading Cache...")

files_downloaded = set()

with open('./files_downloaded_cache.txt', 'r') as cache:
    for line in cache:
        files_downloaded.add(line.replace("\n","")) 
        
print("Loading Data...")

train_data_list = []
validation_data_list = []
test_data_list = [] # TODO

with open('./ground_truth/IIIT-HWS-90K.txt', 'r') as file:
    for line in file:
        # flag: train = 0, validation = 1
        img_path, target, _, train_val_flag = line.split()
        datapoint = {
            'image_path': os.getcwd() + '/Images_90K_Normalized/' + img_path,
            'text': target
        }
        if datapoint['image_path'] not in files_downloaded:
            continue

        if train_val_flag == '0':
            train_data_list.append(datapoint)
        elif train_val_flag == '1':
            validation_data_list.append(datapoint)




print(f'Train size: {len(train_data_list)}; Validation size {len(validation_data_list)}; Test size: {len(test_data_list)}')

# Data Loader

class IIITHWSDataset(Dataset):
    def __init__(self, words, config: DTrOCRConfig):
        super(IIITHWSDataset, self).__init__()
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
    max_position_embeddings = 512
)

train_data = IIITHWSDataset(words=train_data_list, config=config)
validation_data = IIITHWSDataset(words=validation_data_list, config=config)
# test_data = IIITHWSDataset(words=test_data_list, config=config)


train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=mp.cpu_count())
validation_dataloader = DataLoader(validation_data, batch_size=32, shuffle=False, num_workers=mp.cpu_count())
# test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=mp.cpu_count())

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
print(model)

torch.save(model.state_dict(), 'trained_model/pretrained_model_state_dict.pt')

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
optimiser = torch.optim.Adam(params=model.parameters(), lr=1e-4)

EPOCHS = 8
train_losses, train_accuracies = [], []
validation_losses, validation_accuracies = [], []
for epoch in range(EPOCHS):
    epoch_losses, epoch_accuracies = [], []
    for inputs in tqdm.tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch {epoch + 1}'):

        # set gradients to zero
        optimiser.zero_grad()

        # send inputs to same device as model
        inputs = send_inputs_to_device(inputs, device=device)

        # forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            outputs = model(**inputs)

        # calculate gradients
        scaler.scale(outputs.loss).backward()

        # update weights
        scaler.step(optimiser)
        scaler.update()

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

torch.save(model.state_dict(), 'trained_model/trained_model_state_dict.pt')

# Testing

# model.eval()
# model.to('cpu')
# test_processor = DTrOCRProcessor(DTrOCRConfig())



# for test_word_record in test_word_records[:50]:
#     image_file = test_word_record.file_path
#     image = Image.open(image_file).convert('RGB')

#     inputs = test_processor(
#         images=image,
#         texts=test_processor.tokeniser.bos_token,
#         return_tensors='pt'
#     )

#     model_output = model.generate(
#         inputs,
#         test_processor,
#         num_beams=3
#     )

#     predicted_text = test_processor.tokeniser.decode(model_output[0], skip_special_tokens=True)

#     plt.figure(figsize=(10, 5))
#     plt.title(predicted_text, fontsize=24)
#     plt.imshow(np.array(image, dtype=np.uint8))
#     plt.xticks([]), plt.yticks([])
#     plt.savefig("plot.png")