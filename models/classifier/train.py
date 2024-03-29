import os

from tqdm import tqdm

import torch.nn.functional as F
import torch

from models.classifier.evaluate import evaluate_model

def train_model(model, optimizer, dataloader, args, writer, epochs=10):
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-2)
    model.train()
    if args.use_cuda:
        model = model.cuda()
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    cur_epoch = args.cur_epoch
    len_train = epochs * len(dataloader)
    acc = 'no_eval'
    with tqdm(total=len_train) as progress_bar:
        if cur_epoch > 0:
            print('Running from ckpt')
            progress_bar.update(cur_epoch * len(dataloader))
        for e in range(cur_epoch, epochs):
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
                    train_acc=acc,
                    epoch=e+1
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
                acc = evaluate_model(model, dataloader, args, use_tqdm=False)
                model.train()
                progress_bar.set_postfix(
                    train_loss=loss.item(),
                    train_acc=acc,
                    epoch=e+1
                )
                writer.add_scalar('train/acc', acc, e)
    save_loc = os.path.join(args.checkpoint_dir, 'classifier.final.pth')
    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'args': args
    }, save_loc)
    print('Done training, calculating train acc')
    acc = evaluate_model(model, dataloader, args)
    return model, acc
