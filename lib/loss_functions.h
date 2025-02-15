#pragma once

#include <limits>

#include <Eigen/Core>

#include "activations.h"

namespace LossFunctions {

inline constexpr float softmax_cross_entropy(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	return -(references.array() * Activations::softmax(outputs).array().max(std::numeric_limits<float>::epsilon()).log()).sum();
}

inline constexpr float binary_cross_entropy(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	Eigen::MatrixXf clipped = outputs.array().max(std::numeric_limits<float>::epsilon());
	return -(references.array() * clipped.array().log() + (1.f - references.array()) * (1.f - clipped.array()).log()).sum();
}

inline constexpr float mean_squared_error(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	return ((references - outputs).array().square()).colwise().mean().sum();
}

inline constexpr float mean_absolute_error(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	return (references - outputs).array().abs().colwise().mean().sum();
}

}

