#include <Eigen/Dense>

Eigen::VectorXf meanSquaredErrorGradient(const Eigen::MatrixXf &outputs, const Eigen::MatrixXf &targets) {
	return (2 * (outputs - targets)).rowwise().mean();
}

Eigen::VectorXf meanAbsoluteErrorGradient(const Eigen::MatrixXf &outputs, const Eigen::MatrixXf &targets) {
	return ((outputs.array() > targets.array()).cast<float>() - (outputs.array() < targets.array()).cast<float>()).rowwise().mean();
}

Eigen::VectorXf binaryCrossEntropyGradient(const Eigen::MatrixXf &outputs, const Eigen::MatrixXf &targets) {
	// TODO
}

Eigen::VectorXf multiClassCrossEntropyGradient(const Eigen::MatrixXf &outputs, const Eigen::MatrixXf &targets) {
	// TODO
}

Eigen::VectorXf SVMGradient(const Eigen::MatrixXf &outputs, const Eigen::MatrixXf &targets) {
	// TODO
}