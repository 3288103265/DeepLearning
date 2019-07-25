import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

D_in, D_out = 2, 1
MAX_EPOCH = 500

w = torch.tensor([[3.14], [4.56]], requires_grad=True)
def get_data(w):
    x = torch.randn(50, 2, requires_grad=True)
    y = x.mm(w)
    return x, y

x, y = get_data(w)

model = nn.Linear(D_in, D_out, bias=True)
loss_fn = nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(MAX_EPOCH):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)

    if t % 50 == 0:
        print(t, loss.item())

    optimizer.zero_grad()

    optimizer.step()
