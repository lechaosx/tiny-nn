#include <print>
#include <vector>
#include <ranges>
#include <functional>

#include <Eigen/Dense>

#include "activation.h"

struct Layer {
	Eigen::MatrixXf weights;
	Eigen::VectorXf biases;

	struct Activation {
		std::function<Eigen::MatrixXf(const Eigen::MatrixXf &)> activation;
		std::function<Eigen::MatrixXf(const Eigen::MatrixXf &)> derivative;
	} activation;
};

const Layer::Activation Sigmoid { sigmoidActivation, sigmoidDerivative };
const Layer::Activation Tanh { tanhActivation, tanhDerivative };
const Layer::Activation Relu { reluActivation, reluDerivative };


std::vector<Layer> createNeuralNetwork(const std::vector<uint64_t> &topology, const Layer::Activation &activation) {
	std::vector<Layer> nn{};

	for (size_t i = 0; i < std::size(topology) - 1; ++i) {
		nn.emplace_back(Eigen::MatrixXf::Random(topology[i + 1], topology[i]), Eigen::VectorXf::Random(topology[i + 1]), activation);
	}

	return nn;
}

Eigen::MatrixXf feedLayer(const std::vector<Layer> &nn, const Eigen::MatrixXf &inputs, size_t layer = 0) {
	if (layer == std::size(nn)) {
		return inputs;
	}

	return feedLayer(nn, nn[layer].activation.activation((nn[layer].weights * inputs).colwise() + nn[layer].biases), layer + 1);
}

template<typename F>
concept LossDerivative = requires(const F &f, const Eigen::MatrixXf &outputs) {
	{ f(outputs) } -> std::same_as<Eigen::MatrixXf>;
};

template <LossDerivative F>
Eigen::MatrixXf trainLayer(std::vector<Layer> &nn, const Eigen::MatrixXf &inputs, float learningRate, const F &lossDerivative, size_t layer = 0) {
	if (layer == std::size(nn)) {
		return lossDerivative(inputs);
	}

	Eigen::MatrixXf zs = (nn[layer].weights * inputs).colwise() + nn[layer].biases;

	Eigen::MatrixXf delta = trainLayer(nn, nn[layer].activation.activation(zs), learningRate, lossDerivative, layer + 1).cwiseProduct(nn[layer].activation.derivative(zs));

	nn[layer].weights -= learningRate * delta * inputs.transpose();
	nn[layer].biases -= learningRate * delta.rowwise().sum();

	return nn[layer].weights.transpose() * delta;
}

template <LossDerivative F>
void trainNeuralNetwork(std::vector<Layer> &nn, const Eigen::MatrixXf& input, float learningRate, const F &lossDerivative) {
	std::vector<Eigen::MatrixXf> activations, zs;

	activations.push_back(input);

	for (size_t i = 0; i < std::size(nn); ++i) {
		Eigen::MatrixXf z = (nn[i].weights * activations.back()).colwise() + nn[i].biases;
		
		zs.push_back(z);

		activations.push_back(nn[i].activation.activation(z));
	}

	Eigen::MatrixXf error = lossDerivative(activations.back());

	for (int layer = nn.size() - 1; layer >= 0; --layer) {
		Eigen::MatrixXf delta = error.cwiseProduct(nn[layer].activation.derivative(zs[layer]));

		nn[layer].weights -= learningRate * delta * activations[layer].transpose();
		nn[layer].biases -= learningRate * delta.rowwise().sum();

		error = nn[layer].weights.transpose() * delta;
	}
}

int main() {
	std::vector<Layer> nn = createNeuralNetwork({2, 3, 3}, Sigmoid);

	Eigen::MatrixXf inputs(2, 4);
	inputs << 0, 0, 1, 1,
	          0, 1, 0, 1;

	Eigen::MatrixXf targets(3, 4);
	targets << 0, 1, 1, 0,
	           0, 0, 0, 1,
	           0, 1, 1, 1;

	for (size_t epoch = 0; epoch < 100000; ++epoch) {
		auto lossDerivative = [&](const Eigen::MatrixXf &outputs) -> Eigen::MatrixXf {
			
			if (epoch % 1000 == 0) {
				float loss = -(targets.array() * outputs.array().log() + (1 - targets.array()) * (1 - outputs.array()).log()).colwise().sum().mean();
				std::println("epoch {}: loss {}", epoch, loss);
			}
			
			return outputs - targets;
		};

		trainLayer(nn, inputs, 0.1, lossDerivative);
		trainNeuralNetwork(nn, inputs, 0.1, lossDerivative);
	}

	Eigen::MatrixXf outputs = feedLayer(nn, inputs);

	for (size_t i = 0; i < outputs.cols(); ++i) {
		std::println("{} XOR {} = {}", inputs(0, i) > 0.5f, inputs(1, i) > 0.5f, outputs(0, i) > 0.5f);
		std::println("{} AND {} = {}", inputs(0, i) > 0.5f, inputs(1, i) > 0.5f, outputs(1, i) > 0.5f);
		std::println("{} OR {} = {}", inputs(0, i) > 0.5f, inputs(1, i) > 0.5f, outputs(2, i) > 0.5f);
	}

	return 0;
}