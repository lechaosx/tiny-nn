#include "neural_network.h"

#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/core/class_db.hpp>

NeuralNetwork::NeuralNetwork() {
	if (godot::Engine::get_singleton()->is_editor_hint()) {
		set_process_mode(Node::ProcessMode::PROCESS_MODE_DISABLED);
	}
}

void NeuralNetwork::_ready() {
}

void NeuralNetwork::_process(double delta) {
	m_time_passed += m_speed * delta;

	godot::Vector2 new_position = m_initial_position + godot::Vector2(m_amplitude * std::sin(m_time_passed * 2.0), m_amplitude * std::cos(m_time_passed * 1.5));

	m_time_emit += delta;
	while (m_time_emit > 1.0) {
		emit_signal("position_changed", this, new_position);
		m_time_emit -= 1.0;
	}
}

void NeuralNetwork::_bind_methods() {
	godot::ClassDB::bind_method(godot::D_METHOD("get_amplitude"), &NeuralNetwork::get_amplitude);
	godot::ClassDB::bind_method(godot::D_METHOD("set_amplitude", "p_amplitude"), &NeuralNetwork::set_amplitude);
	godot::ClassDB::add_property("NeuralNetwork", godot::PropertyInfo(godot::Variant::FLOAT, "amplitude"), "set_amplitude", "get_amplitude");

	godot::ClassDB::bind_method(godot::D_METHOD("get_speed"), &NeuralNetwork::get_speed);
	godot::ClassDB::bind_method(godot::D_METHOD("set_speed", "p_speed"), &NeuralNetwork::set_speed);
	godot::ClassDB::add_property("NeuralNetwork", godot::PropertyInfo(godot::Variant::FLOAT, "speed", godot::PROPERTY_HINT_RANGE, "0,20,0.01"), "set_speed", "get_speed");

	ADD_SIGNAL(godot::MethodInfo("position_changed", godot::PropertyInfo(godot::Variant::OBJECT, "node"), godot::PropertyInfo(godot::Variant::VECTOR2, "new_pos")));
}