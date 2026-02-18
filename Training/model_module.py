# model_module.py
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from model import BasicBlock, ResNet
from torch.optim.lr_scheduler import OneCycleLR

class ResNet18Lightning(pl.LightningModule):

    def __init__(self, pretrained_path, num_classes=10, lr=1e-3,len_train_loader=204, epochs=10):
        super().__init__()
        self.lr = lr
        self.len_train_loader = len_train_loader
        self.epochs = epochs
        self.model = ResNet(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=10
        )

        state_dict = torch.load(pretrained_path, map_location="cpu")
        self.model.load_state_dict(state_dict)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            steps_per_epoch=self.len_train_loader,
            epochs=self.epochs,
            pct_start=0.3,
            anneal_strategy='linear'
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
