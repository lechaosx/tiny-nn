import conan
import conan.tools.cmake

class GodotCppTestConan(conan.ConanFile):
	settings = "os", "compiler", "build_type", "arch"
	generators = "CMakeDeps", "CMakeToolchain"

	def requirements(self):
		self.requires(self.tested_reference_str)

	def build(self):
		cmake = conan.tools.cmake.CMake(self)
		cmake.configure()
		cmake.build()

	def layout(self):
		conan.tools.cmake.cmake_layout(self)

	def test(self):
		pass