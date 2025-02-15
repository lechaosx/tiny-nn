#pragma once

#include <Eigen/Core>

#include "activations.h"
#include "derivatives.h"

enum struct Activation: uint8_t {
	LINEAR  = 0,
	RELU    = 1,
	SIGMOID = 2,
	TANH    = 3,
};

inline constexpr Eigen::MatrixXf apply_activation(const Eigen::MatrixXf &inputs, Activation activation) {
	switch (activation) {
		case Activation::LINEAR:
			return Activations::linear(inputs);;
		case Activation::RELU:
			return Activations::relu(inputs);
		case Activation::SIGMOID:
			return Activations::sigmoid(inputs);
		case Activation::TANH:
			return Activations::tanh(inputs);
	}

	throw std::runtime_error(std::format("Unknown activation function {}", static_cast<uint8_t>(activation)));
}

inline constexpr Eigen::MatrixXf apply_derivative(const Eigen::MatrixXf &inputs, Activation activation) {
	switch (activation) {
		case Activation::LINEAR:
			return Derivatives::linear(inputs);;
		case Activation::RELU:
			return Derivatives::relu(inputs);
		case Activation::SIGMOID:
			return Derivatives::sigmoid(inputs);
		case Activation::TANH:
			return Derivatives::tanh(inputs);
	}

	throw std::runtime_error(std::format("Unknown activation function {}", static_cast<uint8_t>(activation)));
}

struct Layer {
	Eigen::MatrixXf weights;
	Eigen::VectorXf biases;
	Activation activation;
};

inline constexpr Eigen::MatrixXf feed(std::span<const Layer> nn, const Eigen::MatrixXf &inputs) {
	if (nn.empty()) {
		return inputs;
	}

	const Layer &layer = nn.front();

	return feed(nn.subspan(1), apply_activation((layer.weights * inputs).colwise() + layer.biases, layer.activation));
}

template<typename F>
concept LossDerivative = requires(const F &f, const Eigen::MatrixXf &outputs) {
	{ f(outputs) } -> std::same_as<Eigen::MatrixXf>;
};


template <LossDerivative F>
inline constexpr Eigen::MatrixXf train(std::span<Layer> nn, const Eigen::MatrixXf &inputs, float learningRate, const F &lossDerivative) {
	if (nn.empty()) {
		return lossDerivative(inputs);
	}

	Layer &layer = nn.front();

	auto linear_output = (layer.weights * inputs).colwise() + layer.biases;

	Eigen::MatrixXf delta = train(nn.subspan(1), apply_activation(linear_output, layer.activation), learningRate, lossDerivative).array() * apply_derivative(linear_output, layer.activation).array();

	Eigen::MatrixXf prev_delta = layer.weights.transpose() * delta;

	layer.weights -= learningRate * delta * inputs.transpose() / inputs.cols();
	layer.biases -= learningRate * delta.rowwise().sum() / inputs.cols();

	return prev_delta;
}