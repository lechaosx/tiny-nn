#include <gtest/gtest.h>

#include <serialization.h>

TEST(DeserializeTest, InvalidJsonFormat) {
	nlohmann::json invalid_json;
	invalid_json["weights"] = { {1.0f, 2.0f}, {3.0f, 4.0f} };
	invalid_json["biases"] = { 0.5f, 1.5f };
	
	EXPECT_THROW(deserialize({ invalid_json }), std::exception);
}

TEST(SerializeAndDeserializeTest, RoundTrip) {
	Eigen::MatrixXf weights(2, 2);
	weights << 1.0f, 2.0f, 3.0f, 4.0f;

	Eigen::VectorXf biases(2);
	biases << 0.5f, 1.5f;

	Layer layer(weights, biases, Activation::SIGMOID);

	std::vector<Layer> nn = { layer };

	std::vector<Layer> deserialized_nn = deserialize(serialize(nn));

	ASSERT_EQ(deserialized_nn.size(), nn.size());
	
	const Layer& deserialized_layer = deserialized_nn[0];

	ASSERT_EQ(deserialized_layer.weights.rows(), 2);
	ASSERT_EQ(deserialized_layer.weights.cols(), 2);
	ASSERT_FLOAT_EQ(deserialized_layer.weights(0, 0), 1.0f);
	ASSERT_FLOAT_EQ(deserialized_layer.weights(0, 1), 2.0f);
	ASSERT_FLOAT_EQ(deserialized_layer.weights(1, 0), 3.0f);
	ASSERT_FLOAT_EQ(deserialized_layer.weights(1, 1), 4.0f);

	ASSERT_EQ(deserialized_layer.biases.size(), 2);
	ASSERT_FLOAT_EQ(deserialized_layer.biases(0), 0.5f);
	ASSERT_FLOAT_EQ(deserialized_layer.biases(1), 1.5f);

	ASSERT_EQ(deserialized_layer.activation, Activation::SIGMOID);
}

