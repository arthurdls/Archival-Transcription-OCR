import os

print("Loading Cache...")

files_downloaded = set()

with open('./files_downloaded_cache.txt', 'r') as cache:
    for line in cache:
        files_downloaded.add(line.replace("\n","")) 

print("Updating Cache...")
        
with open('./ground_truth/IIIT-HWS-90K.txt', 'r') as file:
    with open('./files_downloaded_cache.txt', 'a') as cache:
        for line in file:
            # flag: train = 0, validation = 1
            img_path, target, _, train_val_flag = line.split()
            datapoint = {
                'image_path': os.getcwd() + '/Images_90K_Normalized/' + img_path,
                'text': target
            }
            if datapoint['image_path'] not in files_downloaded and os.path.exists(datapoint['image_path']):
                cache.write(datapoint['image_path'] + "\n")