#include <cmath>
#include <algorithm>

inline Eigen::MatrixXf sigmoidActivation(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) { return 1.0f / (1.0f + std::exp(-val)); });
}

inline Eigen::MatrixXf sigmoidDerivative(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) {
		float s = 1.0f / (1.0f + std::exp(-val));
		return s * (1.0f - s);
	});
}

inline Eigen::MatrixXf tanhActivation(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) { return std::tanh(val); });
}

inline Eigen::MatrixXf tanhDerivative(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) { 
		float t = std::tanh(val);
		return 1.0f - t * t; 
	});
}

inline Eigen::MatrixXf reluActivation(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) { return std::max(0.0f, val); });
}

inline Eigen::MatrixXf reluDerivative(const Eigen::MatrixXf& x) {
	return x.unaryExpr([](float val) { return val > 0 ? 1.0f : 0.0f; });
}

inline Eigen::MatrixXf softmaxActivation(const Eigen::MatrixXf& x) {
	Eigen::MatrixXf expVals = (x.rowwise() - x.colwise().maxCoeff()).array().exp();
	return expVals.array().rowwise() / (expVals.colwise().sum()).array();
}