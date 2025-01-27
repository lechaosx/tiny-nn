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
concept LossDerivative = requires(const F &f, const Eigen::VectorXf &outputs) {
	{ f(outputs) } -> std::same_as<Eigen::VectorXf>;
};

template <LossDerivative F>
Eigen::VectorXf trainLayer(std::vector<Layer> &nn, const Eigen::VectorXf &inputs, float learningRate, const F &lossDerivative, size_t layer = 0) {
	if (layer == std::size(nn)) {
		return lossDerivative(inputs);
	}

	Eigen::VectorXf zs = (nn[layer].weights * inputs).colwise() + nn[layer].biases;

	Eigen::VectorXf activations = nn[layer].activation.activation(zs);

	Eigen::VectorXf gradient = nn[layer].activation.derivative(zs).cwiseProduct(trainLayer(nn, activations, learningRate, lossDerivative, layer + 1));

	Eigen::VectorXf delta = learningRate * gradient;

	nn[layer].weights -= delta * inputs.transpose();
	nn[layer].biases -= delta;

	return nn[layer].weights.transpose() * gradient;
}

void trainNeuralNetwork(std::vector<Layer> &nn, const Eigen::VectorXf& input, const Eigen::VectorXf& target, float learningRate) {
	std::vector<Eigen::VectorXf> activations, zs;

	Eigen::VectorXf currentActivation = input;

	activations.push_back(currentActivation);

	for (size_t i = 0; i < std::size(nn); ++i) {
		Eigen::VectorXf z = nn[i].weights * currentActivation + nn[i].biases;
		
		zs.push_back(z);

		currentActivation = nn[i].activation.activation(z);

		activations.push_back(currentActivation);
	}

	Eigen::VectorXf derivative = activations.back() - target;

	for (int layer = nn.size() - 1; layer >= 0; --layer) {
		Eigen::VectorXf gradient = derivative.cwiseProduct(nn[layer].activation.derivative(zs[layer]));

		Eigen::VectorXf delta = gradient * learningRate;

		nn[layer].weights -= delta * activations[layer].transpose();
		nn[layer].biases -= delta;

		derivative = nn[layer].weights.transpose() * gradient;
	}
}

int main() {
	std::vector<Layer> nn = createNeuralNetwork({2, 3, 2}, Sigmoid);

	// Individual sample input and target (XOR problem)
	Eigen::MatrixXf inputs(2, 4);
	inputs << 0, 0, 1, 1,
	          0, 1, 0, 1;

	Eigen::MatrixXf targets(2, 4);
	targets << 0, 1, 1, 0,
	           0, 0, 0, 1;

	// Learning loop
	for (size_t epoch = 0; epoch < 10000; ++epoch) {
		float epochLoss = 0.0f;

		// Train with individual samples
		for (size_t i = 0; i < inputs.cols(); ++i) {
			Eigen::VectorXf inputSample = inputs.col(i);
			Eigen::VectorXf target = targets.col(i);

			// Define the loss derivative function
			auto lossDerivative = [&](const Eigen::VectorXf &output) -> Eigen::VectorXf {
				Eigen::VectorXf diff = output - target;
				//epochLoss += -(target(0) * std::log(output(0)) + (1 - target(0)) * std::log(1 - output(0)));  // Track loss for the epoch
				return diff;
			};

			// Train for the current sample
			//trainLayer(nn, inputSample, 0.1, lossDerivative);
			//trainNeuralNetwork(nn, inputSample, target, 0.1);
		}

		// Print loss for the epoch
		if (epoch % 1000 == 0) {
			std::println("Epoch {}: Loss = {}", epoch, epochLoss / inputs.cols());
		}
	}

	// Final output after training
	Eigen::MatrixXf outputs = feedLayer(nn, inputs);

	// Print results
	for (size_t i = 0; i < outputs.cols(); ++i) {
		std::println("{} XOR {} = {}", inputs(0, i), inputs(1, i), outputs(0, i));
		std::println("{} AND {} = {}", inputs(0, i), inputs(1, i), outputs(1, i));
	}

	return 0;
}