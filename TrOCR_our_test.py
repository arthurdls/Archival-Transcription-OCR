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
from jiwer import cer

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

TrOCRProcessor = TrOCRProcessor.from_pretrained(os.getcwd() + '/pretrained_repos/microsoft/trocr-large-handwritten')
TrOCRModel = VisionEncoderDecoderModel.from_pretrained(os.getcwd() + '/pretrained_repos/microsoft/trocr-large-handwritten')

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

files = {
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

for file_name, test_data_list in files.items():
    with open(f'../single_word/{file_name}/labels.txt', 'r') as file:
        for line in file:
            img_path, target = line.replace(",", "").replace("\n","").split(" ")
            if img_path == 'image':
                continue
            datapoint = {
                'image_path': os.getcwd() + f'/../single_word/{file_name}/' + img_path,
                'text': target
            }
            test_data_list.append(datapoint)

# # Data Loader

class OurDataset(Dataset):
    def __init__(self, words):
        super(OurDataset, self).__init__()
        self.words = words
        self.processor = TrOCRProcessor

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        with Image.open(self.words[item]['image_path']).convert('RGB') as img:
            pixel_values = self.processor(img, return_tensors="pt").pixel_values
        return {
            'pixel_values': pixel_values[0],
            'labels': self.words[item]['text']
        }

files_with_dataloaders = {}

for file_name, test_data_list in files.items():
    test_data = OurDataset(words=test_data_list)
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

model = TrOCRModel
model = torch.compile(model)
model.to(device=device)
print(model)


def evaluate_model(model: torch.nn.Module, processor, dataloader: DataLoader) -> Tuple[float, float]:
    # set model to evaluation mode
    model.eval()

    references, hypotheses = [], []
    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader, total=len(dataloader), desc=f'Evaluating test set'):
            inputs = send_inputs_to_device(inputs, device=0)
            outputs = model.generate(inputs['pixel_values'], max_length = 300)
            hypothesis = processor.batch_decode(outputs, skip_special_tokens=True)

            hypotheses.extend(hypothesis)
            references.extend(inputs['labels'])

    test_cer = cer(reference=references, hypothesis=hypotheses)

    # set model back to training mode
    model.train()

    return test_cer


def send_inputs_to_device(dictionary, device):
    return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in dictionary.items()}


for file_name, test_dataloader in files_with_dataloaders.items():
    test_cer = evaluate_model(model, TrOCRProcessor, test_dataloader)

    print(f"{file_name}:: - CER: {test_cer}")