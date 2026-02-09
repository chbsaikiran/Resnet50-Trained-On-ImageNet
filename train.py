import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor,Compose,Lambda
from model import Bottleneck,ResNet

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

def train(dataloader,model,loss_fn,optimizer):
	size = len(dataloader.dataset)
	model.train()
	for batch, (X,y) in enumerate(dataloader):
		X,y = X.to(device),y.to(device)
		
		pred = model(X)
		loss = loss_fn(pred,y)
		
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		
		if(batch % 100 == 0):
			loss, current = loss.item(), (batch + 1)*len(X)
			print(f"loss: {loss:>7f}  [{current:5d}/{size:5d}")


def test(dataloader,model,loss_fn):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X,y in dataloader:
			X,y = X.to(device),y.to(device)
			pred = model(X)
			test_loss += loss_fn(pred,y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
		test_loss = test_loss/num_batches
		correct = correct/size
		print(f"Test Accuracy : {100*correct:>0.1f} , Avg Test Loss : {test_loss:>8f}")


epochs = 2

for t in range(epochs):
	print(f"Epoch {t+1} ---------------------\n")
	train(train_dataloader,model,loss_fn,optimizer)
	test(test_dataloader,model,loss_fn)

print("Done!!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
