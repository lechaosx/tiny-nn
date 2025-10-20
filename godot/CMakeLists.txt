cmake_minimum_required(VERSION 3.30)

get_target_property(GODOTCPP_SUFFIX godot-cpp GODOTCPP_SUFFIX)

add_custom_target(install_gdextension ALL 
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:neural-network> ${CMAKE_CURRENT_SOURCE_DIR}/neural_network/libneural-network${GODOTCPP_SUFFIX}$<TARGET_FILE_SUFFIX:neural-network>
)
