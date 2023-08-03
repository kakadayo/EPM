import torch

a = torch.tensor([[[0.1, 0.9], [0.5, 0.5]], [[0.5, 0.5], [0.4, 0.6]]])  # 2*2*2
b = torch.tensor([[10., 1.], [1., 10.]])

weighted_score = torch.einsum('bkc,bk->bc', a, b)
print(weighted_score)