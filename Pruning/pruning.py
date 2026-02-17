# pruning.py

import torch
import torch.nn as nn
import torch_pruning as tp


def magnitude_pruner(model, pruning_ratio=0.1):
    example_input = torch.randn(1, 3, 224, 224)

    imp = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = [model.fc]

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_input,
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
    )

    pruner.step()
    return model


def taylor_pruning(model, train_loader, device, pruning_ratio=0.1):

    example_input = torch.randn(1, 3, 224, 224).to(device)
    imp = tp.importance.TaylorImportance()
    ignored_layers = [model.fc]

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_input,
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
    )

    model.train()
    criterion = nn.CrossEntropyLoss()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        if i >= 30:
            break

    pruner.step()
    return model
