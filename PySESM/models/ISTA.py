import torch

def generate_h(l_functions):

  h = torch.rand((l_functions), requires_grad=False)

  for i in range(l_functions):
    if(abs(h[i]) >= 0.75):
        h[i] = 1
    else:
        h[i] = 0
  return h
