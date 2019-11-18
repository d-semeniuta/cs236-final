import torch

def evaluate_model(model, loader, opt):
    device = torch.device('cuda') if opt.use_cuda else torch.device('cpu')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
    return acc
        # print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
