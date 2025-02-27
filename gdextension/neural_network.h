#pragma once

#include <godot_cpp/classes/node.hpp>

class NeuralNetwork: public godot::Node {
	GDCLASS(NeuralNetwork, Node)

public:
	NeuralNetwork();

	void _ready() override;

	void _process(double delta) override;

	void set_amplitude(double amplitude) { m_amplitude = amplitude; }
	double get_amplitude() const { return m_amplitude; }

	void set_speed(double speed) { m_speed = speed; }
	double get_speed() const { return m_speed; }

private:
	static void _bind_methods();

	godot::Vector2 m_initial_position {};

	double m_time_passed = 0.0;
	double m_amplitude = 10.0;
	double m_speed = 2.0;
	double m_time_emit = 0.0;
};