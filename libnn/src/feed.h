#pragma once

#include "nn.h"

struct FeedLayer {
	const Coefficients &coefficients;
	const Activation   &activation;
};

inline constexpr std::vector<FeedLayer> zip(std::span<const Coefficients> coefficients, std::span<const Activation> activations) {
	std::vector<FeedLayer> zipped {};

	for (size_t i = 0; i < std::min({ std::size(coefficients), std::size(activations) }); ++i) {
		zipped.emplace_back(coefficients[i], activations[i]);
	}

	return zipped;
};

inline constexpr Eigen::MatrixXf feed(std::span<const FeedLayer> layers, const Eigen::MatrixXf &inputs) {
	if (layers.empty()) {
		return inputs;
	}

	const FeedLayer &layer = layers.front();

	return feed(layers.subspan(1), layer.activation((layer.coefficients.weights * inputs).colwise() + layer.coefficients.biases));
}