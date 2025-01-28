#include <print>
#include <vector>
#include <span>
#include <array>
#include <ranges>
#include <functional>

#include <Eigen/Core>

#include "activation.h"

struct Activation {
	std::function<Eigen::MatrixXf(const Eigen::MatrixXf &)> activation;
	std::function<Eigen::MatrixXf(const Eigen::MatrixXf &)> derivative;
};

const Activation Sigmoid { sigmoidActivation, sigmoidDerivative };
const Activation Tanh { tanhActivation, tanhDerivative };
const Activation Relu { reluActivation, reluDerivative };

struct Layer {
	Eigen::MatrixXf weights;
	Eigen::VectorXf biases;

	Activation activation;
};

template <size_t N>
std::vector<Layer> constexpr createNeuralNetwork(std::span<const uint64_t, N + 1> topology, std::span<const Activation, N> activations) {
	std::vector<Layer> nn{};

	for (size_t i = 0; i < N; ++i) {
		nn.emplace_back(Eigen::MatrixXf::Random(topology[i + 1], topology[i]), Eigen::VectorXf::Random(topology[i + 1]), activations[i]);
	}

	return nn;
}

Eigen::MatrixXf constexpr feedLayer(std::span<const Layer> nn, const Eigen::MatrixXf &inputs) {
	if (nn.empty()) {
		return inputs;
	}

	const Layer &layer = nn.front();

	return feedLayer(nn.subspan(1), layer.activation.activation((layer.weights * inputs).colwise() + layer.biases));
}

template<typename F>
concept LossDerivative = requires(const F &f, const Eigen::MatrixXf &outputs) {
	{ f(outputs) } -> std::same_as<Eigen::MatrixXf>;
};

template <LossDerivative F>
Eigen::MatrixXf constexpr trainLayer(std::span<Layer> nn, const Eigen::MatrixXf &inputs, float learningRate, const F &lossDerivative) {
	if (nn.empty()) {
		return lossDerivative(inputs);
	}

	Layer &layer = nn.front();

	Eigen::MatrixXf zs = (layer.weights * inputs).colwise() + layer.biases;

	Eigen::MatrixXf delta = trainLayer(nn.subspan(1), layer.activation.activation(zs), learningRate, lossDerivative).cwiseProduct(layer.activation.derivative(zs));

	layer.weights -= learningRate * delta * inputs.transpose();
	layer.biases -= learningRate * delta.rowwise().sum();

	return layer.weights.transpose() * delta;
}

template <LossDerivative F>
void constexpr trainNeuralNetwork(std::span<Layer> nn, const Eigen::MatrixXf& input, float learningRate, const F &lossDerivative) {
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
	std::vector<Layer> nn = createNeuralNetwork<2>(std::array<uint64_t, 3>{2, 3, 3}, std::array<Activation, 2>{Sigmoid, Sigmoid});

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