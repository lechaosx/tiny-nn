#pragma once

#include <fstream>
#include <nlohmann/json.hpp>

#include "nn.h"

nlohmann::json serialize(std::span<const Layer> nn) {
	nlohmann::json model_json {};

	for (const Layer& layer : nn) {
		nlohmann::json layer_json {};
		layer_json["weights"] = std::vector<std::vector<float>>(layer.weights.rows(), std::vector<float>(layer.weights.cols()));

		for (Eigen::Index i = 0; i < layer.weights.rows(); ++i) {
			for (Eigen::Index j = 0; j < layer.weights.cols(); ++j) {
				layer_json["weights"][i][j] = layer.weights(i, j);
			}
		}

		layer_json["biases"] = std::vector<float>(layer.biases.data(),  layer.biases.data() + layer.biases.size());
		layer_json["activation"] = std::format("{}", static_cast<uint8_t>(layer.activation));

		model_json.push_back(layer_json);
	}

	return model_json;
}

std::vector<Layer> deserialize(const nlohmann::json &model_json) {
	std::vector<Layer> nn {};
	
	for (const auto& layer_json : model_json) {
		size_t rows = layer_json["weights"].size();
		size_t cols = layer_json["weights"][0].size();

		Eigen::MatrixXf weights(rows, cols);
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				weights(i, j) = layer_json["weights"][i][j];
			}
		}

		Eigen::VectorXf biases(rows);
		for (size_t i = 0; i < rows; ++i) {
			biases(i) = layer_json["biases"][i];
		}

		Activation activation = static_cast<Activation>(std::stoi(std::string(layer_json["activation"])));

		nn.emplace_back(weights, biases, activation);
	}

	return nn;
}