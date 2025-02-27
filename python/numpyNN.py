import numpy
import json

import common

class Relu:
	index = "1"

	def activation(x):
		return numpy.maximum(0, x)
	
	def derivative(x):
		return (x > 0).astype(float)
	
class Softmax:
	index = "4"

	def activation(x):
		exp_x = numpy.exp(x - numpy.max(x, axis = 1, keepdims = True))
		return exp_x / numpy.sum(exp_x, axis = 1, keepdims = True)
	
	def derivative(x):
		return numpy.ones_like(x)

class Sigmoid:
	index = "2"

	def activation(x):
		return 1 / (1 + numpy.exp(-x))
	
	def derivative(x):
		activated = Sigmoid.activation(x)
		return activated * (1 - activated)
	
activations = {
	"1": Relu,
	"4": Softmax,
	"2": Sigmoid,
}
	
def cross_entropy_loss(y_true, y_pred):
	return -numpy.sum(numpy.sum(y_true * numpy.log(y_pred + 1e-8), axis = 1))

def binary_cross_entropy_loss(y_true, y_pred):
	return -numpy.sum(y_true * numpy.log(y_pred + 1e-8) + (1 - y_true) * numpy.log(1 - y_pred + 1e-8))

def binary_cross_entropy_derivative(y_true, y_pred):
	clipped = numpy.maximum(y_pred, numpy.finfo(float).eps)
	return -(y_true / clipped) + ((1.0 - y_true) / (1.0 - clipped))

def one_hot_encode(labels, num_classes = 10):
	return numpy.eye(num_classes)[labels]

def xavier_init(size):
	in_size, out_size = size
	limit = numpy.sqrt(6.0 / (in_size + out_size))
	return numpy.random.uniform(-limit, limit, size=size)

class Layer:
	def __init__(self):
		self.weights = None
		self.biases = None
		self.activation = None

def serialize_layer(layer):
	return {
		"weights": layer.weights.astype(numpy.float32).T.tolist() if layer.weights is not None else None,
		"biases": layer.biases.astype(numpy.float32).flatten().tolist() if layer.biases is not None else None,
		"activation": layer.activation.index if layer.activation is not None else None
	}

def deserialize_layer(data):
	layer = Layer()
	layer.weights = numpy.array(data["weights"]).astype(numpy.float32).T
	layer.biases = numpy.array(data["biases"]).astype(numpy.float32).reshape(1, -1)
	layer.activation = activations[data["activation"]]
	return layer

def serialize(layers):
	return [serialize_layer(layer) for layer in layers]

def deserialize(datas):
	return [deserialize_layer(data) for data in datas]

def xavier_layer(inputs, outputs, activation):
	layer = Layer()
	layer.weights = xavier_init((inputs, outputs))
	layer.biases = numpy.zeros((1, outputs))
	layer.activation = activation

	return layer

nn = [
	xavier_layer(common.input_size, common.hidden_size1, Relu),
	xavier_layer(common.hidden_size1, common.hidden_size2, Relu),
	xavier_layer(common.hidden_size2, common.output_size, Softmax),
]

def feed(nn, inputs):
	if not nn:
		return inputs
	
	layer = nn[0]
	
	return feed(nn[1:], layer.activation.activation(numpy.dot(inputs, layer.weights) + layer.biases))

def train(nn, inputs, loss_derivative):
	if not nn:
		return loss_derivative(inputs)
	
	layer = nn[0]

	linear_output = numpy.matmul(inputs, layer.weights) + layer.biases

	delta = train(nn[1:], layer.activation.activation(linear_output), loss_derivative) * layer.activation.derivative(linear_output)

	prev_delta = numpy.matmul(delta, layer.weights.T)

	layer.weights -= numpy.matmul(inputs.T, delta)
	layer.biases  -= numpy.sum(delta, axis = 0, keepdims = True)

	return prev_delta

# Training loop
for epoch in range(common.epochs):
	loss = 0

	for images, labels in common.train_loader:
		# Flatten the image data and convert to numpy array
		images = images.view(-1, 28 * 28).numpy()
		labels = common.one_hot_encode(labels.numpy())

		def loss_derivative(x):
			global loss
			loss += common.cross_entropy_loss(labels, x)
			return (x - labels) * common.learning_rate / len(x)

		train(common.nn, images, loss_derivative)

	print(f"Epoch {epoch + 1}/{common.epochs}, Loss: {loss / len(common.train_dataset):.4f}")

# Evaluation
correct = 0
total = 0

for images, labels in common.test_loader:
	images = images.view(-1, 28 * 28).numpy()
	labels = labels.numpy()

	predictions = numpy.argmax(feed(common.nn, images), axis = 1)

	correct += (predictions == labels).sum()
	total += labels.size

print(f"Test Accuracy: {100 * correct / total:.2f}%")

with open("output.json", "w") as outfile: 
	json.dump(common.serialize(common.nn), outfile)
