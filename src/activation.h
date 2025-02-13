#include <cmath>

#include <Eigen/Core>

namespace Activations {

inline constexpr Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) { return 1.0f / (1.0f + std::exp(-val)); });
}

inline constexpr Eigen::MatrixXf sigmoid_derivative(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) {
		float s = 1.0f / (1.0f + std::exp(-val));
		return s * (1.0f - s);
	});
}

inline constexpr Eigen::MatrixXf tanh(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) { return std::tanh(val); });
}

inline constexpr Eigen::MatrixXf tanh_derivative(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) { 
		float t = std::tanh(val);
		return 1.0f - t * t; 
	});
}

inline constexpr Eigen::MatrixXf relu(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) {
		return std::max(0.0f, val); 
	});
}

inline constexpr Eigen::MatrixXf relu_derivative(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) {
		return val > 0.f ? 1.0f : 0.0f; 
	});
}

inline constexpr Eigen::MatrixXf softmax(const Eigen::MatrixXf& x) {
	Eigen::MatrixXf expVals = (x.array().rowwise() - x.array().colwise().maxCoeff()).exp();
	return expVals.array().rowwise() / expVals.array().colwise().sum();
}

inline constexpr Eigen::MatrixXf softmax_derivative(const Eigen::MatrixXf& x) {
	return Eigen::MatrixXf::Ones(x.rows(), x.cols()); // Asssume cross-entropy loss
}

}