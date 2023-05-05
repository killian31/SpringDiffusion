import sys

import matplotlib.pyplot as plt
import requests
import torch

sys.path.append("../")

from scripts.sample_image import sample_save_image
from utils.util import get_loss


def train(optimizer, epochs, device, dataloader, batch_size, T, model, img_size, betas, use_colab=False):
    if not use_colab:
        from tqdm import tqdm
    else:
        from tqdm.notebook import tqdm
    ntfy_name = f"train_spring_{epochs}_{batch_size}_{T}_{img_size}"
    print(f"Sending notifications to ntfy.sh/{ntfy_name}")
    requests.post(
        f"https://ntfy.sh/{ntfy_name}",
        data="Training has started".encode(encoding="utf-8"),
        headers={"Title": "Training Status", "Priority": "2"},
    )
    try:
        model.to(device)
        losses = []
        for epoch in range(epochs):
            batch_losses = []
            with pbar as enumerate(dataloader)
                for step, batch in pbar:
                    optimizer.zero_grad()

                    t = torch.randint(0, T, (batch_size,), device=device).long()
                    loss = get_loss(model, batch[0], t, betas, device)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                    pbar.set_description(
                        f"Epoch {epoch} | step {step:03d} Loss: {loss.item()}"
                    )

            sample_save_image(model, betas, epoch, img_size, device, T, use_colab)
            losses.append(sum(batch_losses) / len(batch_losses))
            requests.post(
                f"https://ntfy.sh/{ntfy_name}",
                data=f"Epoch {epoch} | Loss: {losses[epoch]}".encode(encoding="utf-8"),
                headers={"Title": "Training Status", "Priority": "2"},
            )
            plt.figure(figsize=(12, 16))
            plt.plot(losses)
            plt.savefig(f"./Current_losses{img_size}.jpg")
            requests.put(
                f"https://ntfy.sh/{ntfy_name}",
                data=open(f"./Current_losses{img_size}.jpg", "rb"),
                headers={"Filename": f"Current_losses{img_size}.jpg"},
            )
            plt.close()

        torch.save(model.state_dict(), f"./weights/weights{img_size}.pt")
        plt.figure(figsize=(12, 16))
        plt.plot(losses)
        plt.savefig(f"./losses{img_size}.png")

        return model, losses

    except KeyboardInterrupt:
        print(f"Training interrupted. Saving weights in ./weights/weights{img_size}.pt")
        torch.save(model.state_dict(), f"./weights/weights{img_size}.pt")
        plt.figure(figsize=(12, 16))
        plt.plot(losses)
        plt.savefig(f"./losses{img_size}.png")

        return model, losses
