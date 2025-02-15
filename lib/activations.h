#pragma once

#include <cmath>

#include <Eigen/Core>

namespace Activations {

inline constexpr Eigen::MatrixXf linear(const Eigen::MatrixXf& x) {
	return x;
}

inline constexpr Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) { return 1.0f / (1.0f + std::exp(-val)); });
}

inline constexpr Eigen::MatrixXf tanh(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) { return std::tanh(val); });
}

inline constexpr Eigen::MatrixXf relu(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) {
		return std::max(0.0f, val); 
	});
}

inline constexpr Eigen::MatrixXf softmax(const Eigen::MatrixXf& x) {
	Eigen::MatrixXf expVals = (x.array().rowwise() - x.array().colwise().maxCoeff()).exp();
	return expVals.array().rowwise() / expVals.array().colwise().sum();
}

}