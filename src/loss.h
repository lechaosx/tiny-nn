#include <limits>

#include <Eigen/Core>

inline constexpr float cross_entropy(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	return -(references.array() * outputs.array().max(std::numeric_limits<float>::epsilon()).log()).sum();
}

inline constexpr Eigen::MatrixXf cross_entropy_derivative(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	return outputs - references;
}

inline constexpr float binary_cross_entropy(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	Eigen::MatrixXf clipped = outputs.array().max(std::numeric_limits<float>::epsilon());
	
	return -(references.array() * clipped.array().log() + (1.f - references.array()) * (1.f - clipped.array()).log()).sum();
}

inline constexpr Eigen::MatrixXf binary_cross_entropy_derivative(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	Eigen::MatrixXf clipped = outputs.array().max(std::numeric_limits<float>::epsilon());

	return -(references.array() / clipped.array()) + ((1.f - references.array()) / (1.f - clipped.array()));
}

inline constexpr float mean_squared_error(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	return ((references - outputs).array().square()).colwise().mean().sum();
}

inline constexpr Eigen::MatrixXf mean_squared_error_derivative(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	return 2 * (outputs - references) / references.rows();
}

inline constexpr float mean_absolute_error(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	return (references - outputs).array().abs().colwise().mean().sum();
}

inline constexpr Eigen::MatrixXf mean_absolute_error_derivative(const Eigen::MatrixXf& outputs, const Eigen::MatrixXf& references) {
	return (outputs - references).array().sign();
}