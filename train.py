import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor,Compose,Lambda
from torch.utils.tensorboard import SummaryWriter
from model import Bottleneck,ResNet
from math import sqrt
import time

transform = Compose([
    ToTensor(),                 # (1, H, W)
    Lambda(lambda x: x.repeat(3, 1, 1))  # → (3, H, W)
])

train_data = datasets.FashionMNIST(root="data",train="True",download="True",transform=transform,)
test_data = datasets.FashionMNIST(root="data",train="False",download="True",transform=transform,)

batch_size = 64

train_dataloader = DataLoader(train_data,batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

for X,y in test_dataloader:
	print(f"shape of X [N,C,H,W]:{X.shape}")
	print(f"shape of y : {y.shape} {y.dtype}")
	break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device} device")


model = ResNet(Bottleneck, [3, 4, 6, 3],10).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)

def train(dataloader, model, loss_fn, optimizer, epoch, writer):
    size = len(dataloader.dataset)
    model.train()
    start0 = time.time()
    start = time.time()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        batch_size = len(X)
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], {(current/size * 100):>4f}%")
            step = epoch * size + current
            writer.add_scalar('training loss',
                            loss,
                            step)
            new_start = time.time()
            delta = new_start - start
            start = new_start
            if batch != 0:
                print("Done in ", delta, " seconds")
                remaining_steps = size - current
                speed = 100 * batch_size / delta
                remaining_time = remaining_steps / speed
                print("Remaining time (seconds): ", remaining_time)
        optimizer.zero_grad()
    print("Entire epoch done in ", time.time() - start0, " seconds")


def test(dataloader, model, loss_fn, epoch, writer, train_dataloader, calc_acc5=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, correct_top5 = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if calc_acc5:
                _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
                correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()
    test_loss /= num_batches
    step = epoch * len(train_dataloader.dataset)
    if writer != None:
        writer.add_scalar('test loss',
                            test_loss,
                            step)
    correct /= size
    correct_top5 /= size
    if writer != None:
        writer.add_scalar('test accuracy',
                            100*correct,
                            step)
        if calc_acc5:
            writer.add_scalar('test accuracy5',
                            100*correct_top5,
                            step)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if calc_acc5:
        print(f"Test Error: \n Accuracy-5: {(100*correct_top5):>0.1f}%, Avg loss: {test_loss:>8f} \n")


writer = SummaryWriter('runs/' + "training_fashionmnist")
epochs = 2

for t in range(epochs):
	print(f"Epoch {t+1} ---------------------\n")
	train(train_dataloader, model, loss_fn, optimizer, epoch=t, writer=writer)
	test(test_dataloader, model, loss_fn, t + 1, writer, train_dataloader=train_dataloader, calc_acc5=True)

print("Done!!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
