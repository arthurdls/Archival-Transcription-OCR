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

# Model

print("Loading Model...")

if torch.cuda.is_available():
    device = torch.device("cuda")  
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

torch.set_float32_matmul_precision('high')

state_dict = torch.load('./trained_model/epoch_7_checkpoint_model_state_dict.pt', weights_only=True)
config = DTrOCRConfig(
    gpt2_hf_model = os.getcwd() + '/pretrained_repos/openai-community/gpt2',
    vit_hf_model = os.getcwd() + '/pretrained_repos/google/vit-base-patch16-244',
    max_position_embeddings = 512
)
model = DTrOCRLMHeadModel(config)
model = torch.compile(model)
model.to(device=device)
model.load_state_dict(state_dict)

print(model)

# Freeze patch_embeddings, token_embedding, and positional_embedding
for name, param in model._orig_mod.transformer.named_parameters():
    if "patch_embeddings" in name or "token_embedding" in name or "positional_embedding" in name:
        param.requires_grad = False

# Freeze all hidden transformer layers except the last one
for i, layer in enumerate(model._orig_mod.transformer.hidden_layers):
    if i < len(model._orig_mod.transformer.hidden_layers) - 1:  # Freeze all layers except the last one
        for param in layer.parameters():
            param.requires_grad = False
    else:  # Unfreeze the last layer
        for param in layer.parameters():
            param.requires_grad = True

# Ensure that the language model head is trainable
for param in model._orig_mod.language_model_head.parameters():
    param.requires_grad = True
# Now only the last transformer block and the language model head will be trained

# Iterate over all parameters in the model and print those with requires_grad=True (trainable)
for name, param in model._orig_mod.named_parameters():
    if param.requires_grad:
        print(f"Trainable parameter: {name}")

# Data

print("Loading Data...")

from datasets import load_from_disk
IAM = load_from_disk(os.getcwd() + "/IAM") # "Teklia/IAM-line")

train_data_list = IAM['train']
validation_data_list = IAM['validation']
test_data_list = IAM['test']

print(f'Train size: {len(train_data_list)}; Validation size {len(validation_data_list)}; Test size: {len(test_data_list)}')

# Data Loader

class IAMDataset(Dataset):
    def __init__(self, words, config: DTrOCRConfig):
        super(IAMDataset, self).__init__()
        self.words = words
        self.processor = DTrOCRProcessor(config, add_eos_token=True, add_bos_token=True)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        inputs = self.processor(
            images=self.words[item]['image'].convert('RGB'),
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


train_data = IAMDataset(words=train_data_list, config=config)
validation_data = IAMDataset(words=validation_data_list, config=config)
test_data = IAMDataset(words=test_data_list, config=config)


train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=mp.cpu_count())
validation_dataloader = DataLoader(validation_data, batch_size=32, shuffle=False, num_workers=mp.cpu_count())
# test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=mp.cpu_count())


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

EPOCHS = 50
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

torch.save(model.state_dict(), 'trained_model/finetuned_model_state_dict.pt')

# Testing


# model.load_state_dict(torch.load('./epoch_7_checkpoint_model_state_dict.pt', weights_only=True))
# model.eval()
# model.to('cpu')
# test_processor = DTrOCRProcessor(config)


# for datapoint in IAM['test']:
#     image = datapoint['image']
#     actual_text = datapoint['text']

#     inputs = test_processor(
#         images=image.convert('RGB'),
#         texts=test_processor.tokeniser.bos_token,
#         return_tensors='pt'
#     )

#     model_output = model.generate(
#         inputs,
#         test_processor,
#         num_beams=3
#     )

#     predicted_text = test_processor.tokeniser.decode(model_output[0], skip_special_tokens=True)
#     print(f"Actual: {actual_text}, Predicted: {predicted_text}")

#     plt.figure(figsize=(10, 5))
#     plt.title(predicted_text, fontsize=24)
#     plt.imshow(np.array(image, dtype=np.uint8))
#     plt.xticks([]), plt.yticks([])
#     plt.savefig("plots/plot.png")