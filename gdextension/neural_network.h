#pragma once

#include <nn.h>

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/string.hpp>

class NeuralNetwork: public godot::Node {
	GDCLASS(NeuralNetwork, Node)

public:
	NeuralNetwork();

	bool load_coefficients(const godot::String &path);
	godot::TypedArray<float> feed(const godot::TypedArray<float> &inputs);

private:
	static void _bind_methods();

	std::vector<Coefficients> m_coefficients;
	std::vector<Activation>   m_activations;
};

