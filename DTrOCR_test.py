import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from DTrOCR.dtrocr.processor import DTrOCRProcessor
from DTrOCR.dtrocr.config import DTrOCRConfig
from DTrOCR.dtrocr.model import DTrOCRLMHeadModel

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

print("Loading Model...")

if torch.cuda.is_available():
    device = torch.device("cuda")  
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

torch.set_float32_matmul_precision('high')

# download repos for offline use
config = DTrOCRConfig(
    # attn_implementation='flash_attention_2'
    # gpt2_hf_model = os.getcwd() + '/pretrained_repos/openai-community/gpt2',
    # vit_hf_model = os.getcwd() + '/pretrained_repos/google/vit-base-patch16-244',
    max_position_embeddings = 512
)

model = DTrOCRLMHeadModel(config)
model = torch.compile(model)
model.to(device=device)
print(model)

model.load_state_dict(torch.load('example.pt', weights_only=True))
model.eval()
model.to('cpu')
test_processor = DTrOCRProcessor(config)

test_data = []

with open(f'../washingtondb-v1.0/ground_truth/labeled_transcribed_lines.txt', 'r') as file:
    for line in file:
        split_line = line.replace("\n","").split(" ")
        img_name, target = split_line[0], " ".join(split_line[1:])
        datapoint = {
            'image_path': os.getcwd() + f'/../washingtondb-v1.0/data/line_images_normalized/{img_name}.png',
            'text': target
        }
        test_data.append(datapoint)

for datapoint in test_data:
    with Image.open(datapoint['image_path']).convert('RGB') as img:
        image = img
    actual_text = datapoint['text']

    inputs = test_processor(
        images=image,
        texts=test_processor.tokeniser.bos_token,
        return_tensors='pt'
    )

    model_output = model.generate(
        inputs,
        test_processor,
        num_beams=3
    )

    predicted_text = test_processor.tokeniser.decode(model_output[0], skip_special_tokens=True)
    print(f"Actual: {actual_text} \nPredicted: {predicted_text}")
    with open('model_lines_results.txt', 'a') as f:
        f.write(f'{actual_text} > {predicted_text}\n')

    ### Visual Test ###
    # plt.figure(figsize=(10, 5))
    # plt.title(predicted_text, fontsize=24)
    # plt.imshow(np.array(image, dtype=np.uint8))
    # plt.xticks([]), plt.yticks([])
    # plt.savefig("plots/plot.png")