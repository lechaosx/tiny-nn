#include <functional>

#include <Eigen/Core>

struct Activation {
	std::function<Eigen::MatrixXf(const Eigen::MatrixXf &)> activation;
	std::function<Eigen::MatrixXf(const Eigen::MatrixXf &)> derivative;
};

struct Layer {
	Eigen::MatrixXf weights;
	Eigen::VectorXf biases;
	Activation activation;
};

inline constexpr Eigen::MatrixXf feed(std::span<const Layer> nn, const Eigen::MatrixXf &inputs) {
	if (nn.empty()) {
		return inputs;
	}

	const Layer &layer = nn.front();

	return feed(nn.subspan(1), layer.activation.activation((layer.weights * inputs).colwise() + layer.biases));
}

template<typename F>
concept LossDerivative = requires(const F &f, const Eigen::MatrixXf &outputs) {
	{ f(outputs) } -> std::same_as<Eigen::MatrixXf>;
};


template <LossDerivative F>
inline constexpr Eigen::MatrixXf train(std::span<Layer> nn, const Eigen::MatrixXf &inputs, float learningRate, const F &lossDerivative) {
	if (nn.empty()) {
		return lossDerivative(inputs);
	}

	Layer &layer = nn.front();

	auto linear_output = (layer.weights * inputs).colwise() + layer.biases;

	Eigen::MatrixXf delta = train(nn.subspan(1), layer.activation.activation(linear_output), learningRate, lossDerivative).array() * layer.activation.derivative(linear_output).array();

	Eigen::MatrixXf prev_delta = layer.weights.transpose() * delta;

	layer.weights -= learningRate * delta * inputs.transpose() / inputs.cols();
	layer.biases -= learningRate * delta.rowwise().sum() / inputs.cols();

	return prev_delta;
}