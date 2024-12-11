from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os
from os import listdir
import jiwer
from datasets import load_dataset

IMG_FOLDER_PATH = '' # change to folder containing images
GROUND_TRUTH_TEXT_FILE_PATH = '' # change to newline separated textfile of ground truths


device = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten').to(device)

def get_image_hypotheses(processor, model, images):
    hypotheses = []
    i = 0
    for image in images:
        if i % 100 == 0:
            print(i)
        i += 1
        image = image.convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        hypothesis_ids = model.generate(pixel_values, max_length = 150)
        hypothesis_text = processor.batch_decode(hypothesis_ids, skip_special_tokens=True)[0]
        hypotheses.append(hypothesis_text)
    return hypotheses

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

    pixel_values = processor(
        images=image,
        return_tensors='pt'
    ).pixel_values

    model_output = model.generate(pixel_values)

    predicted_text = processor.batch_decode(model_output[0], skip_special_tokens=True)
    print(f"Actual: {actual_text} \nPredicted: {predicted_text}")
    with open('model_lines_results.txt', 'a') as f:
        f.write(f'{actual_text} > {predicted_text}\n')
