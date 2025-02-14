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
const Activation Softmax { Activations::softmax, Activations::softmax_derivative };

struct Layer {
	Eigen::MatrixXf weights;
	Eigen::VectorXf biases;
	Activation activation;
};

Layer random_layer(Eigen::Index inputs, Eigen::Index outputs, const Activation &activation) {
	return { Eigen::MatrixXf::Random(outputs, inputs), Eigen::VectorXf::Zero(outputs), activation };
}

constexpr Eigen::MatrixXf feed(std::span<const Layer> nn, const Eigen::MatrixXf &inputs) {
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
constexpr Eigen::MatrixXf train(std::span<Layer> nn, const Eigen::MatrixXf &inputs, float learningRate, const F &lossDerivative) {
	if (nn.empty()) {
		return lossDerivative(inputs);
	}

	Layer &layer = nn.front();

	Eigen::MatrixXf activations = layer.activation.activation((layer.weights * inputs).colwise() + layer.biases);

	Eigen::MatrixXf delta = train(nn.subspan(1), activations, learningRate, lossDerivative).array() * layer.activation.derivative(activations).array();

	Eigen::MatrixXf prev_delta = layer.weights.transpose() * delta;

	layer.weights -= learningRate * delta * inputs.transpose() / inputs.cols();
	layer.biases -= learningRate * delta.rowwise().sum() / inputs.cols();

	return prev_delta;
}

int main(int argc, const char *argv[]) {
	if (argc != 5)
	{
		std::println("Usage: {} <path-to-train-inputs> <path-to-train-labels> <path-to-test-inputs> <path-to-test-labels>", argv[0]);
		return 1;
	}

	Eigen::MatrixXf inputs = read_idx_images(argv[1]);
	Eigen::MatrixXf labels = read_idx_labels(argv[2]);

	Eigen::MatrixXf test_inputs = read_idx_images(argv[3]);
	Eigen::MatrixXf test_labels = read_idx_labels(argv[4]);

	std::vector<Layer> nn {
		random_layer(inputs.rows(), 512, Relu),
		random_layer(512, 256, Relu),
		random_layer(256, labels.rows(), Softmax),
	};

	const size_t BATCH_SIZE = 64;

	for (size_t epoch = 0; epoch < 10; ++epoch) {
		for (Eigen::Index batch_start = 0; batch_start < inputs.cols(); batch_start += BATCH_SIZE) {
			const size_t batch_end = std::min(batch_start + BATCH_SIZE, static_cast<size_t>(inputs.cols()));

			Eigen::MatrixXf batch_inputs = inputs.block(0, batch_start, inputs.rows(), batch_end - batch_start);
			Eigen::MatrixXf batch_labels = labels.block(0, batch_start, labels.rows(), batch_end - batch_start);
			
			auto lossDerivative = [&](const Eigen::MatrixXf &outputs) -> Eigen::MatrixXf {
				return outputs - batch_labels;
			};

			train(nn, batch_inputs, 0.01, lossDerivative);
		}

		Eigen::MatrixXf outputs = feed(nn, test_inputs);

		int correct_predictions = 0;

		for (int i = 0; i < outputs.cols(); ++i) {
			int predicted_class;
			int expected_class;

			outputs.col(i).maxCoeff(&predicted_class);
			test_labels.col(i).maxCoeff(&expected_class);

			correct_predictions += (predicted_class == expected_class);
		}

		float accuracy = static_cast<float>(correct_predictions) / outputs.cols() * 100;

		std::println("accuracy {} %", accuracy);
	}

	// TODO serialize architecture and coeffs, load in different tool

	return 0;
}