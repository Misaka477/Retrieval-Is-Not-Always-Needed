"""Convert our .npy data to NanoGPT's .bin format"""
import numpy as np, pickle, os

data = np.load('checkpoints/mohe_fw_rwkv_1b.npy', mmap_mode='r')
N = len(data)
split = int(N * 0.9)

train = data[:split].astype(np.uint16)
val = data[split:].astype(np.uint16)

os.makedirs('nanoGPT/data/rina', exist_ok=True)
train.tofile('nanoGPT/data/rina/train.bin')
val.tofile('nanoGPT/data/rina/val.bin')

meta = {'vocab_size': 65536, 'tokenizer': 'rwkv'}
with open('nanoGPT/data/rina/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print(f'Train: {len(train)/1e6:.0f}M tokens')
print(f'Val:   {len(val)/1e6:.0f}M tokens')
