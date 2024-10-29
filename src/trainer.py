import torch
import numpy as np
import os

import matplotlib.pyplot as plt
from IPython.display import clear_output


class Trainer:
    def __init__(self, config):

        self.checkpoint_path = config["checkpoint_path"]
        self.checkpoint = config["checkpoint"]

        self.model = config["model"]
        self.loss_fn = config["loss_fn"]
        self.optimizer = config["optim"]
        self.num_epochs = config["epochs"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        self.avg_loss_min = np.Inf
        self.last_epoch = 1

    def restore_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.avg_loss_min = checkpoint["avg_loss"]
            self.last_epoch = checkpoint["epoch"]
            # future proof for loading a generic model
            info = checkpoint["info"]

            print(f"Last epoch: {self.last_epoch}")
            print(f"Avg loss min: {self.avg_loss_min}")

    def train_epoch(self, train_loader, epoch, plot=False):

        model = self.model
        optimizer = self.optimizer

        model.train()

        losses = []
        progresses = []

        plt.ion()

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()

            output = model(data)

            loss = self.loss_fn(output, target)

            loss.backward()

            optimizer.step()

            batch_size = data.shape[0]
            if batch_idx % batch_size == 0:
                sample_idx = batch_idx * len(data)
                num_samples = len(train_loader.dataset)
                num_batches = len(train_loader)
                progress = 100.0 * batch_idx / num_batches

                losses.append(loss.item())
                progresses.append(progress)

                if plot:
                    clear_output(wait=True)
                    plt.figure(figsize=(10, 6))
                    plt.plot(progresses, losses, label="Loss")
                    plt.xlabel("Progress (%)")
                    plt.ylabel("Loss")
                    plt.title(f"Epoch {epoch}")
                    plt.legend()
                    plt.grid(True)
                    plt.show()

                print(
                    f"Train Epoch: {epoch} [s {sample_idx+1}/{num_samples} b {batch_idx+1}/{num_batches} ({progress:.0f}%)]\tLoss: {loss.item():.6f}"
                )

    def test(self, loader):
        """
        returns the average loss on the provided set
        """

        model = self.model

        model.eval()
        avg_loss = 0
        correct = 0

        for data, target in loader:
            batch_size = data.shape[0]
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            avg_loss += self.loss_fn(output, target, reduction="sum").item()

            pred = output.argmax(dim=1, keepdim=True)

            # sanity check
            pred = pred.view(batch_size)  # [bs,]
            target = target.view(batch_size)  # [bs,]

            # compute prediction ok
            batch_pred_ok = pred.eq(target).sum().item()
            correct += batch_pred_ok

        avg_loss /= len(loader.dataset)
        num_samples = len(loader.dataset)
        accuracy = 100.0 * correct / num_samples
        print(f"Eval average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}%")
        return avg_loss, accuracy

    def val(self, loader, epoch, avg_loss_min):
        """
        saves the model if the new avg loss is less than the provided minimum,
        returns the current minimum average loss
        """

        avg_loss, accuracy = self.test(loader)

        if avg_loss <= avg_loss_min:

            print(
                f"Validation loss decreased: {avg_loss_min} ----> {avg_loss} , Accuracy: {accuracy:.3f}% , Saving Model..."
            )

            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "avg_loss": avg_loss,
                    "epoch": epoch,
                    "info": self.info,
                },
                self.checkpoint_path,
            )

            return avg_loss

        return avg_loss_min

    def train_loop(self, train_loader, val_loader, plot=False):
        for epoch in range(self.last_epoch, self.num_epochs + 1):
            self.train_epoch(train_loader, epoch, plot)
            avg_loss_min = self.val(val_loader, epoch, avg_loss_min)
