#pragma once

#include "nn.h"

#include <span>

struct TrainLayer {
	Coefficients         &coefficients;
	Activation           &activation;
	ActivationDerivative &activation_derivative;
};

inline std::vector<TrainLayer> zip(std::span<Coefficients> coefficients, std::span<Activation> activations, std::span<ActivationDerivative> activation_derivatives) {
	std::vector<TrainLayer> zipped {};

	for (size_t i = 0; i < std::min({ std::size(coefficients), std::size(activations), std::size(activation_derivatives) }); ++i) {
		zipped.emplace_back(coefficients[i], activations[i], activation_derivatives[i]);
	}

	return zipped;
}

inline Eigen::MatrixXf train(std::span<const TrainLayer> layers, const Eigen::MatrixXf &inputs, const LossDerivative &lossDerivative) {
	if (layers.empty()) {
		return lossDerivative(inputs);
	}

	const TrainLayer &layer = layers.front();

	Eigen::MatrixXf linear_output = (layer.coefficients.weights * inputs).colwise() + layer.coefficients.biases;

	Eigen::MatrixXf delta = train(layers.subspan(1), layer.activation(linear_output), lossDerivative).array() * layer.activation_derivative(linear_output).array();

	Eigen::MatrixXf prev_delta = layer.coefficients.weights.transpose() * delta;

	layer.coefficients.weights -= delta * inputs.transpose();
	layer.coefficients.biases -= delta.rowwise().sum();

	return prev_delta;
}