# train.py

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from data_module import ImageNetDataModule
from model_module import ResNet18Lightning


def main():

    WNID_TO_LABEL = {
        "n02077923": 0,
        "n02058221": 1,
        "n02051845": 2,
        "n02037110": 3,
        "n02028035": 4,
        "n01440764": 5,
        "n01443537": 6,
        "n01484850": 7,
        "n01491361": 8,
        "n01494475": 9,
    }

    data_module = ImageNetDataModule(
        train_dir="./../imagenet10/train.X",
        val_dir="./../imagenet10/val.X",
        wnid_map=WNID_TO_LABEL,
        batch_size=64
    )

    model = ResNet18Lightning(
        pretrained_path="./../resnet18_imagenet10.pth",
        num_classes=10,
        lr=0.0007626985859023446e-05,
        len_train_loader=204,
        epochs=10
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, datamodule=data_module)

    torch.save(model, "resnet18_imagenet10_full.pth")
    torch.save(model.state_dict(), "resnet18_imagenet10.pth")


if __name__ == "__main__":
    main()
