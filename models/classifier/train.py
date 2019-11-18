from tqdm import tqdm

import torch.nn.functional as F
import torch

import pdb

from models.classifier.evaluate import evaluate_model

def train_model(model, dataloader, opt, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-2)

    if opt.use_cuda:
        model = model.cuda()  # move the model parameters to CPU/GPU
    device = torch.device('cuda') if opt.use_cuda else torch.device('cpu')

    len_train = epochs * len(dataloader)
    with tqdm(total=len_train) as progress_bar:
        for e in range(epochs):
            for t, (x, y) in enumerate(dataloader):
                model.train()  # put model to training mode
                x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)

                scores = model(x)
                loss = F.cross_entropy(scores, y)

                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                optimizer.zero_grad()

                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                loss.backward()

                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                optimizer.step()
                progress_bar.update(1)
            acc = evaluate_model(model, dataloader, opt)
            progress_bar.set_postfix(train_acc=acc)
    return model, acc
