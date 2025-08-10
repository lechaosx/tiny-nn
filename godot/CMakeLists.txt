cmake_minimum_required(VERSION 3.30)

add_custom_target(install_gdextension ALL 
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:neural-network> ${CMAKE_CURRENT_SOURCE_DIR}/neural_network/libneural-network.$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>.$<LOWER_CASE:$<CONFIG>>.${CMAKE_SYSTEM_PROCESSOR}$<TARGET_FILE_SUFFIX:neural-network>
)
