import sys

import torch
from tqdm import tqdm

sys.path.append("../")

from utils.util import get_loss, sample_save_image


def train(optimizer, epochs, device, dataloader, batch_size, T, model, img_size, betas):
    model.to(device)
    losses = []
    for epoch in range(epochs):
        pbar = tqdm(enumerate(dataloader))
        batch_losses = []
        for step, batch in pbar:
            optimizer.zero_grad()

            t = torch.randint(0, T, (batch_size,), device=device).long()
            loss = get_loss(model, batch[0], t, device)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            pbar.set_description(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()}")
        losses.append(sum(batch_losses) / len(batch_losses))
        sample_save_image(model, betas, epoch, img_size, device, T)

    return model, losses
