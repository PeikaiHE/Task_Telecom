import torch
import os
import matplotlib.pyplot as plt
from utils.biased_mnist import get_biased_mnist_dataloader, SimpleConvNet
import numpy as np
import wandb


def extract_samples(model, data_loader, device):
    model.eval()
    bias_target_aligned = []
    bias_target_misaligned = []

    with torch.no_grad():
        for data, target, biased_target, index in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)

            # to test the model
            # print(f"Predictions: {preds}")
            # print(f"Targets: {target}")

            for i in range(data.size(0)):
                if preds[i] == target[i]:
                    bias_target_aligned.append(data[i])
                else:
                    bias_target_misaligned.append(data[i])

                # if we have enough samples
                if len(bias_target_aligned) >= 10 and len(bias_target_misaligned) >= 10:
                    break

            if len(bias_target_aligned) >= 10 and len(bias_target_misaligned) >= 10:
                break

    print(f"Aligned samples: {len(bias_target_aligned)}, Misaligned samples: {len(bias_target_misaligned)}")
    return bias_target_aligned[:10], bias_target_misaligned


def plot_samples(bias_target_aligned, bias_target_misaligned):
    fig, axs = plt.subplots(2, 10, figsize=(15, 3))

    for i in range(10):
        # normalize to [0, 1]
        aligned_img = bias_target_aligned[i].detach().cpu().numpy()

        misaligned_img = bias_target_misaligned[i].detach().cpu().numpy()
        misaligned_img = misaligned_img * 0.5 + 0.5

        # plot the aligned sample
        axs[0, i].imshow(np.clip(aligned_img.transpose(1, 2, 0), 0, 1))
        axs[0, i].axis('off')

        # plot the misaligned sample
        axs[1, i].imshow(np.clip(misaligned_img.transpose(1, 2, 0), 0, 1))
        axs[1, i].axis('off')

    axs[0, 0].set_ylabel('Bias-Target\nAligned', fontsize=12)
    axs[1, 0].set_ylabel('Bias-Target\nMisaligned', fontsize=12)

    plt.show()
    # But here we have some bugs, IDK


def main(args):
    # load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleConvNet().to(device)  # use SimpleConvNet
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # load the test data
    test_loader = get_biased_mnist_dataloader(
        root='./data',
        batch_size=64,
        data_label_correlation=args.rho,
        n_confusing_labels=9,
        train=False
    )

    print("Test data done")

    # extract samples
    bias_target_aligned, bias_target_misaligned = extract_samples(model, test_loader, device)

    if bias_target_aligned is not None:
        print("bias_target_aligned is not None")
    elif bias_target_misaligned is not None:
        print("bias_target_misaligned is not None")
    # plot samples
    plot_samples(bias_target_aligned, bias_target_misaligned)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract and plot samples from a trained model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model .pth file.")
    parser.add_argument('--rho', type=float, default=0.99, help="Data label correlation for BiasedMNIST.")
    parser.add_argument('--log', action='store_true', help="Use wandb logging.")
    parser.add_argument('--project', type=str, default='biased-mnist', help="wandb project name.")

    args = parser.parse_args()

    if args.log:
        wandb.init(config={}, project=args.project)
        args = wandb.config

    main(args)
