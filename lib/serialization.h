#pragma once

#include <fstream>
#include <nlohmann/json.hpp>

#include "nn.h"

nlohmann::json serialize(std::span<const Coefficients> layers) {
	nlohmann::json model_json {};

	for (const Coefficients& coefficients: layers) {
		nlohmann::json layer_json {};
		layer_json["weights"] = std::vector<std::vector<float>>(coefficients.weights.rows(), std::vector<float>(coefficients.weights.cols()));

		for (Eigen::Index i = 0; i < coefficients.weights.rows(); ++i) {
			for (Eigen::Index j = 0; j < coefficients.weights.cols(); ++j) {
				layer_json["weights"][i][j] = coefficients.weights(i, j);
			}
		}

		layer_json["biases"] = std::vector<float>(coefficients.biases.data(),  coefficients.biases.data() + coefficients.biases.size());

		model_json.push_back(layer_json);
	}

	return model_json;
}

std::vector<Coefficients> deserialize(const nlohmann::json &model_json) {
	std::vector<Coefficients> layers {};
	
	for (const auto& coefficients_json : model_json) {
		size_t rows = coefficients_json["weights"].size();
		size_t cols = coefficients_json["weights"][0].size();

		Eigen::MatrixXf weights(rows, cols);
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				weights(i, j) = coefficients_json["weights"][i][j];
			}
		}

		Eigen::VectorXf biases(rows);
		for (size_t i = 0; i < rows; ++i) {
			biases(i) = coefficients_json["biases"][i];
		}

		layers.emplace_back(weights, biases);
	}

	return layers;
}