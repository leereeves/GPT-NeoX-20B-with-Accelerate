import torch

d = torch.load('./cache/layer_02.pt')

for key,value in d.items():
    print(key)