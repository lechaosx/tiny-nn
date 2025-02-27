import torch

import common

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NN(torch.nn.Module):
	def __init__(self):
		super(NN, self).__init__()
		self.fc1 = torch.nn.Linear(common.input_size, common.hidden_size1)
		self.fc2 = torch.nn.Linear(common.hidden_size1, common.hidden_size2)
		self.fc3 = torch.nn.Linear(common.hidden_size2, common.output_size)
	
	def forward(self, x):
		x = x.view(-1, common.input_size)
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		return self.fc3(x)


model = NN().to(device)
criterion = torch.nn.CrossEntropyLoss()

parameters = list(model.parameters())

for epoch in range(common.epochs):
	model.train()
	
	total_loss = 0

	for images, labels in common.train_loader:
		images, labels = images.to(device), labels.to(device)

		for param in parameters:
			if param.grad is not None:
				param.grad.zero_()
		
		output = model(images)
		
		loss = criterion(output, labels)
		
		loss.backward()
		
		with torch.no_grad():
			for param in parameters:
				param -= common.learning_rate * param.grad
		
		total_loss += loss.item()

	print(f"Epoch {epoch + 1}/{common.epochs}, Loss: {total_loss / len(common.train_loader):.4f}")

model.eval()

correct = 0

total = 0

with torch.no_grad():
	for images, labels in common.test_loader:

		images, labels = images.to(device), labels.to(device)

		output = model(images)

		predictions = torch.argmax(output, dim = 1)

		correct += (predictions == labels).sum().item()

		total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
