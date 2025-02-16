#include <print>

#include <Eigen/Core>

#include <nn.h>
#include <loss_functions.h>
#include <serialization.h>

#include "idx.h"

Layer xavier_layer(Eigen::Index inputs, Eigen::Index outputs, const Activation &activation) {
	float limit = std::sqrt(6.f / (inputs + outputs));

	return { Eigen::MatrixXf::Random(outputs, inputs) * limit, Eigen::VectorXf::Zero(outputs), activation };
}

constexpr Eigen::MatrixXf one_hot_encode(const Eigen::RowVector<uint8_t, Eigen::Dynamic> &labels, uint8_t num_classes) {
	Eigen::MatrixXf one_hot = Eigen::MatrixXf::Zero(num_classes, labels.cols());
	
	for (int i = 0; i < labels.cols(); ++i) {
		one_hot(labels(i), i) = 1.f;
	}

	return one_hot;
}

int main(int argc, const char *argv[]) {
	if (argc != 5)
	{
		std::println("Usage: {} <path-to-train-inputs> <path-to-train-labels> <path-to-test-inputs> <path-to-test-labels>", argv[0]);
		return 1;
	}

	Eigen::MatrixXf inputs = read_idx_images(argv[1]);
	Eigen::RowVector<uint8_t, Eigen::Dynamic> labels = read_idx_labels(argv[2]);

	uint8_t num_classes = *std::max_element(labels.data(), labels.data() + labels.cols()) + 1;

	std::vector<Layer> nn {
		xavier_layer(inputs.rows(), 512, Activation::RELU),
		xavier_layer(512, 256, Activation::RELU),
		xavier_layer(256, num_classes, Activation::SOFTMAX),
	};

	const size_t NUM_EPOCHS = 10;
	const size_t BATCH_SIZE = 64;

	for (size_t epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
		float loss = 0.f;

		for (Eigen::Index batch_start = 0; batch_start < inputs.cols(); batch_start += BATCH_SIZE) {
			const size_t batch_end = std::min(batch_start + BATCH_SIZE, static_cast<size_t>(inputs.cols()));

			Eigen::MatrixXf batch_inputs = inputs.block(0, batch_start, inputs.rows(), batch_end - batch_start);
			Eigen::MatrixXf batch_labels = one_hot_encode(labels.block(0, batch_start, labels.rows(), batch_end - batch_start), num_classes);
			
			auto lossDerivative = [&](const Eigen::MatrixXf &outputs) -> Eigen::MatrixXf {
				loss += LossFunctions::softmax_cross_entropy(outputs, batch_labels);
				return Derivatives::softmax_cross_entropy(outputs, batch_labels) * 0.01f / outputs.cols(); // Also applies learning rate
			};

			train(nn, batch_inputs, lossDerivative);
		}

		std::println("Epoch {}/{}, Loss: {}", epoch + 1, NUM_EPOCHS, loss / inputs.cols());
	}

	Eigen::MatrixXf test_inputs = read_idx_images(argv[3]);
	Eigen::RowVector<uint8_t, Eigen::Dynamic> test_labels = read_idx_labels(argv[4]);

	Eigen::MatrixXf outputs = feed(nn, test_inputs);

	int correct_predictions = 0;

	for (int i = 0; i < outputs.cols(); ++i) {
		int predicted_class;
		outputs.col(i).maxCoeff(&predicted_class);
		correct_predictions += (predicted_class == test_labels(i));
	}

	float accuracy = static_cast<float>(correct_predictions) / outputs.cols() * 100;

	std::println("Accuracy {} %", accuracy);

	std::ofstream output("output.json");
	output << serialize(nn);

	return 0;
}