#pragma once

#include <cmath>

#include <Eigen/Core>

#include "activations.h"

namespace Derivatives {

inline constexpr Eigen::MatrixXf linear(const Eigen::MatrixXf& x) {
	return Eigen::MatrixXf::Ones(x.rows(), x.cols());
}

inline constexpr Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) {
		float s = 1.0f / (1.0f + std::exp(-val));
		return s * (1.0f - s);
	});
}

inline constexpr Eigen::MatrixXf tanh(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) { 
		float t = std::tanh(val);
		return 1.0f - t * t; 
	});
}

inline constexpr Eigen::MatrixXf relu(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) {
		return val > 0.f ? 1.0f : 0.0f; 
	});
}

inline constexpr Eigen::MatrixXf softmax_cross_entropy(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	return Activations::softmax(outputs) - references;
}

inline constexpr Eigen::MatrixXf binary_cross_entropy(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	Eigen::MatrixXf clipped = outputs.array().max(std::numeric_limits<float>::epsilon());

	return -(references.array() / clipped.array()) + ((1.f - references.array()) / (1.f - clipped.array()));
}

inline constexpr Eigen::MatrixXf mean_squared_error(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	return 2 * (outputs - references) / references.rows();
}

inline constexpr Eigen::MatrixXf mean_absolute_error(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	return (outputs - references).array().sign();
}

}