#include <print>
#include <vector>
#include <span>
#include <array>
#include <ranges>
#include <functional>

#include <Eigen/Core>

#include "activation.h"
#include "idx.h"

struct Activation {
	std::function<Eigen::MatrixXf(const Eigen::MatrixXf &)> activation;
	std::function<Eigen::MatrixXf(const Eigen::MatrixXf &)> derivative;
};

const Activation Sigmoid { Activations::sigmoid, Activations::sigmoid_derivative };
const Activation Tanh { Activations::tanh, Activations::tanh_derivative };
const Activation Relu { Activations::relu, Activations::relu_derivative };

struct Layer {
	Eigen::MatrixXf weights;
	Eigen::VectorXf biases;

	Activation activation;
};

Layer random_layer(size_t inputs, size_t outputs, const Activation &activation) {
	return { Eigen::MatrixXf::Random(outputs, inputs), Eigen::VectorXf::Random(outputs), activation };
}

Eigen::MatrixXf constexpr feed(std::span<const Layer> nn, const Eigen::MatrixXf &inputs) {
	if (nn.empty()) {
		return inputs;
	}

	const Layer &layer = nn.front();

	return feed(nn.subspan(1), layer.activation.activation((layer.weights * inputs).colwise() + layer.biases));
}

template<typename F>
concept LossDerivative = requires(const F &f, const Eigen::MatrixXf &outputs) {
	{ f(outputs) } -> std::same_as<Eigen::MatrixXf>;
};

template <LossDerivative F>
Eigen::MatrixXf constexpr train(std::span<Layer> nn, const Eigen::MatrixXf &inputs, float learningRate, const F &lossDerivative) {
	if (nn.empty()) {
		return lossDerivative(inputs);
	}

	Layer &layer = nn.front();

	Eigen::MatrixXf zs = (layer.weights * inputs).colwise() + layer.biases;

	Eigen::MatrixXf delta = train(nn.subspan(1), layer.activation.activation(zs), learningRate, lossDerivative).cwiseProduct(layer.activation.derivative(zs));

	layer.weights -= learningRate * delta * inputs.transpose();
	layer.biases -= learningRate * delta.rowwise().sum();

	return layer.weights.transpose() * delta;
}

int main(int argc, const char *argv[]) {
	if (argc != 3)
	{
		std::println("Usage: {} <path-to-inputs> <path-to-labels>", argv[0]);
		return 1;
	}

	Eigen::MatrixXf inputs = read_idx_images(argv[1]);
	Eigen::MatrixXf labels = read_idx_labels(argv[2]);

	std::vector<Layer> nn {
		random_layer(inputs.rows(), 700, Sigmoid),
		random_layer(700, 100, Sigmoid),
		random_layer(100, labels.rows(), Sigmoid),
	};

	const size_t BATCH_SIZE = 32;

	for (size_t epoch = 0; epoch < 10; ++epoch) {
		for (size_t batch_start = 0; batch_start < inputs.cols(); batch_start += BATCH_SIZE) {
			size_t batch_end = std::min(batch_start + BATCH_SIZE, static_cast<size_t>(inputs.cols()));

			Eigen::MatrixXf batch_inputs = inputs.block(0, batch_start, inputs.rows(), batch_end - batch_start);
			Eigen::MatrixXf batch_labels = labels.block(0, batch_start, labels.rows(), batch_end - batch_start);
			
			auto lossDerivative = [&](const Eigen::MatrixXf &outputs) -> Eigen::MatrixXf {
				return outputs - batch_labels; // TODO softmax
			};

			train(nn, batch_inputs, 0.01, lossDerivative);
		}

		Eigen::MatrixXf outputs = feed(nn, inputs); // TODO use testing dataset
		float loss = -(labels.array() * outputs.array().log() + (1 - labels.array()) * (1 - outputs.array()).log()).colwise().sum().mean();
		std::println("epoch {}: loss {}", epoch, loss); // TODO softmax
	}

	// TODO serialize architecture and coeffs, load in different tool

	Eigen::MatrixXf outputs = feed(nn, inputs);

	// TODO print output, use testing dataset

	return 0;
}