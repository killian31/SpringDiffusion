import sys

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

sys.path.append("../")

from scripts.sample_image import sample_save_image
from utils.util import get_loss


def train(optimizer, epochs, device, dataloader, batch_size, T, model, img_size, betas):
    try:
        model.to(device)
        losses = []
        for epoch in range(epochs):
            pbar = tqdm(enumerate(dataloader), unit="batch")
            batch_losses = []
            with tqdm(dataloader, unit="batch") as pbar:
                for step, batch in enumerate(pbar):
                    optimizer.zero_grad()

                    t = torch.randint(0, T, (batch_size,), device=device).long()
                    loss = get_loss(model, batch[0], t, betas, device)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                    pbar.set_description(
                        f"Epoch {epoch} | step {step:03d} Loss: {loss.item()}"
                    )

            sample_save_image(model, betas, epoch, img_size, device, T)
            losses.append(sum(batch_losses) / len(batch_losses))

        torch.save(model.state_dict(), "./weights/weights.pt")
        plt.figure(figsize=(12, 16))
        plt.plot(losses)
        plt.savefig("./losses.png")

        return model, losses

    except KeyboardInterrupt:
        print("Training interrupted. Saving weights in ./weights/weights.pt")
        torch.save(model.state_dict(), "./weights/weights.pt")
        plt.figure(figsize=(12, 16))
        plt.plot(losses)
        plt.savefig("./losses.png")

        return model, losses
