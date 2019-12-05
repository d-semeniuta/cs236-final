import os

from tqdm import tqdm

import torch.nn.functional as F
import torch

from models.classifier.evaluate import evaluate_model

def train_model(model, optimizer, dataloader, args, writer, epochs=10):
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-2)

    if args.use_cuda:
        model = model.cuda()
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')

    len_train = epochs * len(dataloader)
    with tqdm(total=len_train) as progress_bar:
        for e in range(epochs):
            for t, (x, y) in enumerate(dataloader):
                model.train()
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.long)

                scores = model(x)
                loss = F.cross_entropy(scores, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress_bar.update(1)
                progress_bar.set_postfix(
                    train_loss=loss.item(),
                    epoch=e
                )
                step = e*len(dataloader) + t
                writer.add_scalar('train/loss', loss.item(), step)
            if e % args.save_every_class == 0:
                save_loc = os.path.join(args.checkpoint_dir, 'classifier.last.pth')
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'args': args
                }, save_loc)
            if e % args.eval_every_class == 0:
                acc = evaluate_model(model, dataloader, args)
                progress_bar.set_postfix(
                    train_loss=loss.item(),
                    train_acc=acc
                    epoch=e
                )
                writer.add_scalar('train/acc', acc, epoch)

    print('Done training, calculating train acc')
    acc = evaluate_model(model, dataloader, args)
    return model, acc
