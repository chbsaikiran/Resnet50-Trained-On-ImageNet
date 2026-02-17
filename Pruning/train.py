# train.py

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from config import *
from datamodule import ImageNetDataModule
from lightning_module import PruningDistillationModule
from model import BasicBlock,Bottleneck,ResNet


def load_full_model(path, device):
    model = ResNet(
        block=BasicBlock,
        layers=[2,2,2,2],
        num_classes=10
    )

    #model = torch.load(path, map_location='cpu')
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)

    model.to(device)
    return model

def load_any_model(path, device):

    ckpt = torch.load(path, map_location="cpu")

    model_class = ckpt["model_class"]
    model_kwargs = ckpt["model_kwargs"]

    model = model_class(**model_kwargs)
    model.load_state_dict(ckpt["state_dict"])

    model.to(device)
    return model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = load_full_model(TEACHER_PATH, device)
    student = load_full_model(TEACHER_PATH, device)

    wnid_map = {
        "n02077923": 0,  # sea lion
        "n02058221": 1,  # albatross
        "n02051845": 2,  # pelican
        "n02037110": 3,  # oystercatcher
        "n02028035": 4,  # redshank
        "n01440764": 5,  # tench
        "n01443537": 6,  # goldfish
        "n01484850": 7,  # great white shark
        "n01491361": 8,  # tiger shark
        "n01494475": 9,  # hammerhead
    }

    datamodule = ImageNetDataModule(
        train_dir="./../imagenet10/train.X",
        val_dir="./../imagenet10/val.X",
        wnid_map=wnid_map,
        batch_size=BATCH_SIZE
    )

    model = PruningDistillationModule(
        teacher=teacher,
        student=student,
        lr=LR
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        #precision="16-mixed",  # AMP
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, datamodule=datamodule)

    torch.save(student,"./../temp.pth")


if __name__ == "__main__":
    main()
