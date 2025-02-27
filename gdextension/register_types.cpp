#include "neural_network.h"

#include <gdextension_interface.h>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/core/class_db.hpp>

static void initialize_module(godot::ModuleInitializationLevel p_level) {
	if (p_level == godot::MODULE_INITIALIZATION_LEVEL_SCENE) {
		godot::ClassDB::register_class<NeuralNetwork>();
	}
}

extern "C" GDExtensionBool GDE_EXPORT init(GDExtensionInterfaceGetProcAddress p_get_proc_address, const GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
	godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

	init_obj.register_initializer(initialize_module);
	init_obj.set_minimum_library_initialization_level(godot::MODULE_INITIALIZATION_LEVEL_SCENE);

	return init_obj.init();
}
