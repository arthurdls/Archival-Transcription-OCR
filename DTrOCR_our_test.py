import os
import sys
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
from jiwer import cer

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

# # Data Loader

class OurDataset(Dataset):
    def __init__(self, words, config: DTrOCRConfig):
        super(OurDataset, self).__init__()
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
    gpt2_hf_model = os.getcwd() + '/pretrained_repos/openai-community/gpt2',
    vit_hf_model = os.getcwd() + '/pretrained_repos/google/vit-base-patch16-244',
    max_position_embeddings = 512
)

files_with_dataloaders = {}

for file_name, test_data_list in files.items():
    test_data = OurDataset(words=test_data_list, config=config)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=mp.cpu_count())
    files_with_dataloaders[file_name] = test_dataloader

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
model.load_state_dict(torch.load('./trained_model/retrain_old_singleword_model/epoch_16_checkpoint_new_synthetic_model_state_dict.pt', weights_only=True, map_location=torch.device(device)))
model.to(device=device)
# print(model)

def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, processor: DTrOCRProcessor) -> Tuple[float, float]:
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

# for file_name, test_data_list in files.items():
file_name = sys.argv[1] # list(files.keys())[int(sys.argv[1])] # python file argument for which file to run
print("Working on:", file_name)

test_data_list = files[file_name]
model.eval()
model.to('cpu')
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
    f.write(f"out\n")


# for file_name, test_dataloader in files_with_dataloaders.items():
#     test_loss, test_accuracy = evaluate_model(model, test_dataloader, DTrOCRProcessor(config))
#     print(f"{file_name}:: Loss: {test_loss} - Accuracy: {test_accuracy}")
