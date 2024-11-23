"""
print the location of the necessary repos for the model to train offline
"""

from huggingface_hub import snapshot_download

print(snapshot_download(repo_id="openai-community/gpt2"))
print(snapshot_download(repo_id="google/vit-base-patch16-224"))
