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
import jiwer

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

TrOCRProcessor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
TrOCRModel = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.multiprocessing.set_sharing_strategy('file_system')

FOLDER_PATH = '../synthetic_data'    # change to folder holding folders of images and ground truth text file

test_files = {f:[] for f in os.listdir(FOLDER_PATH)}

for file_name, test_data_list in test_files.items():
    with open(f'{FOLDER_PATH}/{file_name}/labels.txt', 'r') as file:
        for line in file:
            s = line.replace(",", "").split(" ")
            img_path, target = s[0], " ".join(s[1:])
            if img_path == 'image':
                continue
            datapoint = {
                'image_path': f'{FOLDER_PATH}/{file_name}/' + img_path,
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

test_files_with_dataloaders = {}

for file_name, test_data_list in test_files.items():
    test_data = OurDataset(words=test_data_list)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=mp.cpu_count())
    test_files_with_dataloaders[file_name] = test_dataloader

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
model.to(device=device)
print(model)


def evaluate_model(model: torch.nn.Module, processor, dataloader: DataLoader) -> Tuple[float, float]:
    # set model to evaluation mode
    model.eval()

    references, hypotheses = [], []
    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader, total=len(dataloader), desc=f'Evaluating test set'):
            inputs = send_inputs_to_device(inputs, device=device)
            outputs = model.generate(inputs['pixel_values'], max_length = 300)
            hypothesis = processor.batch_decode(outputs, skip_special_tokens=True)

            hypotheses.extend(hypothesis)
            references.extend(inputs['labels'])

    accuracy = jiwer.cer(reference=references, hypothesis=hypotheses)

    # set model back to training mode
    model.train()

    return accuracy

def send_inputs_to_device(dictionary, device):
    return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in dictionary.items()}


if __name__ == '__main__':
    for file_name, test_dataloader in test_files_with_dataloaders.items():
        test_accuracy = evaluate_model(model, TrOCRProcessor, test_dataloader)

        print(f"{file_name}:: - cer: {test_accuracy}")