import torch

from tqdm import tqdm

def evaluate_model(model, loader, args):
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    num_correct = 0
    num_samples = 0
    model.eval()
    with tqdm(total=len(loader)) as progress_bar:
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.long)
                scores = model(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
                progress_bar.update(1)
            acc = float(num_correct) / num_samples
    return acc
