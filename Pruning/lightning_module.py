# lightning_module.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
from losses import distillation_loss
from pruning import taylor_pruning
from torch.optim.lr_scheduler import OneCycleLR


class PruningDistillationModule(pl.LightningModule):

    def __init__(self, teacher, student, epochs=10,lr=1e-3,pruning_ratio=0.3,len_train_loader = 10):
        super().__init__()

        self.teacher = teacher.eval()
        self.student = student
        self.epochs = epochs
        self.lr = lr
        self.pruning_ratio = pruning_ratio
        self.len_train_loader = len_train_loader

    def forward(self, x):
        return self.student(x)

    def on_fit_start(self):
        # Taylor pruning before training
        device = self.device
        train_loader = self.trainer.datamodule.train_dataloader()
        
        taylor_pruning(self.student, train_loader, device,self.pruning_ratio)

    def training_step(self, batch, batch_idx):
        x, y = batch

        with torch.no_grad():
            teacher_logits = self.teacher(x)

        student_logits = self.student(x)
        loss = distillation_loss(student_logits, teacher_logits, y)

        preds = torch.argmax(student_logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.student(x)
        loss = nn.CrossEntropyLoss()(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.student.parameters(),
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


        return [optimizer], [scheduler]
