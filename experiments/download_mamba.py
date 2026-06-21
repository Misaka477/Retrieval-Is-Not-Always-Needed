"""Download Mamba 130M teacher"""
from huggingface_hub import snapshot_download
snapshot_download('state-spaces/mamba-130m-hf', cache_dir='models/teacher')
print('Done')
