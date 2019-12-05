import argparse

def get_cgan_args():
    parser = argparse.ArgumentParser('CGAN Training')
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument("--save_every", type=int, default=25, help="interval between saving the model")
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints', help="Checkpoint directory")
    parser.add_argument("--run_name", required=True, type=str)
    parser.add_argument("--use_cuda", action='store_true', help="Use CUDA if available")
    parser.add_argument("--load_checkpoint", action='store_true', help="Run from checkpoint")
    args = parser.parse_args()
    if args.use_cuda:
        args.use_cuda = torch.cuda.is_available()
    return args
