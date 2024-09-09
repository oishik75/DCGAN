import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator, Generator, initialize_weights

def train(generator, discriminator, dataloader, device, args):
    # Create optimizers
    opt_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn((32, args.z_dim, 1, 1)).to(device) # For evaluation

    # Create summary writers to store generated images during evaluation
    writer = SummaryWriter(args.logs_dir + "/" + args.exp_name)
    step=0

    generator.train()
    discriminator.train

    for epoch in tqdm(range(args.n_epochs)):
        for batch_idx, (real, _) in tqdm(enumerate(dataloader)):
            # Train Discriminator: max log(D(x)) + log(1-D(G(z)))
            # Real
            real = real.to(device)
            disc_real_output = discriminator(real).reshape(-1) # Shape: batch_size
            loss_disc_real = -torch.log(disc_real_output).mean() # -log(D(x)) since we are maximizing
            # Generated
            noise = torch.randn((args.batch_size, args.z_dim, 1, 1)).to(device)
            disc_generated_output = discriminator(generator(noise)).reshape(-1)
            loss_disc_generated = -torch.log(1 - disc_generated_output).mean()
            loss_disc = loss_disc_real + loss_disc_generated
            discriminator.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator: min log(1-D(G(z))) <---> max log(D(G(z)))
            disc_generated_output = discriminator(generator(noise)).reshape(-1)
            loss_gen = -torch.log(disc_generated_output).mean()
            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Add loss to tensorboard
            writer.add_scalar("generator_loss", loss_gen, epoch * len(dataloader) + batch_idx)
            writer.add_scalar("disciminator_loss", loss_disc, epoch * len(dataloader) + batch_idx)

            # Print to tensorboard after every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{args.n_epochs}] Batch {batch_idx}/{len(dataloader)} Loss D: {loss_disc:.4f} Loss G: {loss_gen:.4f}")
                with torch.no_grad():
                    generated = generator(fixed_noise)
                    # Get upto 32 samples
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_gen = torchvision.utils.make_grid(generated[:32], normalize=True)

                    writer.add_image("Real", img_grid_real, global_step=step)
                    writer.add_image("Generated", img_grid_gen, global_step=step)

                step += 1

        # Save Model
        torch.save({
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict()
        }, f"{args.save_dir}/{args.exp_name}/epoch_{epoch}.pt")




def main():
    parser = argparse.ArgumentParser("DCGAN Training Arguments")
    # DATASET ARGUMENTS
    parser.add_argument("--dataset_name", choices=["MNIST", "FashionMNIST"], help="Dataset Name")
    parser.add_argument("--data_path", default=None, help="Dataset folder path")
    # HYPERPARAMETERS
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--image_channels", type=int, default=3)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--features_d", type=int, default=64)
    parser.add_argument("--features_g", type=int, default=64)
    # OTHERS
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--save_dir", default="models")

    args = parser.parse_args()
    print(args)

    assert args.dataset_name or args.data_path, "Either dataset_name or dataset_path must be provided."

    if args.exp_name is None:
        args.exp_name = args.dataset_name if args.dataset_name else args.data_path.replace("/", "_")

    # Create save directory if it does not exist
    os.makedirs(f"{args.save_dir}/{args.exp_name}", exist_ok = True)

    # Create Transforms
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(args.image_channels)],
                [0.5 for _ in range(args.image_channels)]
            )
        ]
    )

    # Load Dataset
    if args.dataset_name == "MNIST":
        dataset = datasets.MNIST(root="data/", train=True, transform=transform, download=True)
    elif args.dataset_name == "FashionMNIST":
        dataset = datasets.FashionMNIST(root="data/", train=True, transform=transform, download=True)
    else:
        dataset = datasets.ImageFolder(root=args.data_path, transform=transform)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize Models
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    generator = Generator(args.z_dim, args.image_channels, args.features_g).to(device)
    initialize_weights(generator)
    discriminator = Discriminator(args.image_channels, args.features_d).to(device)
    initialize_weights(discriminator)

    train(generator, discriminator, loader, device, args)


if __name__ == "__main__":
    main()