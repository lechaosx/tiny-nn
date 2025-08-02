from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMakeDeps, CMakeToolchain, CMake
from conan.tools.scm import Git
from conan.tools.files import copy, collect_libs
import os

class GodotCppRecipe(ConanFile):
	name = "godot-cpp"
	package_type = "library"
	
	settings = "os", "compiler", "build_type", "arch"
	
	options = {
		"shared": [True, False],
		"fPIC": [True, False],
		"hot_reload": [True, False],
	}
	
	default_options = {
		"shared": False,
		"fPIC": True,
		"hot_reload": True,
	}

	def source(self):
		git = Git(self)
		git.clone(url="https://github.com/godotengine/godot-cpp.git", target=".", args=["--recursive", f"--branch {self.version}"])

	def config_options(self):
		if self.settings.os == "Windows":
			del self.options.fPIC

	def configure(self):
		if self.options.shared:
			del self.options.fPIC

	def layout(self):
		cmake_layout(self)

	def generate(self):
		deps = CMakeDeps(self)
		deps.generate()
		
		tc = CMakeToolchain(self)
		
		if self.options.hot_reload:
			tc.preprocessor_definitions["HOT_RELOAD_ENABLED"] = None
			
			if str(self.settings.compiler) in ["gcc", "clang"]:
				tc.extra_cxxflags.append("-fno-gnu-unique")
		
		tc.generate()

	def build(self):
		cmake = CMake(self)
		cmake.configure()
		cmake.build()

	def package(self):
		copy(self, "*.h", os.path.join(self.source_folder, "gdextension"), os.path.join(self.package_folder, "include"))
		copy(self, "*.hpp", os.path.join(self.source_folder, "include"), os.path.join(self.package_folder, "include"))
		copy(self, "*.hpp", os.path.join(self.build_folder, "gen", "include"), os.path.join(self.package_folder, "include"))
		copy(self, "*.inc", os.path.join(self.source_folder, "include"), os.path.join(self.package_folder, "include"))
		copy(self, "*.inc", os.path.join(self.build_folder, "gen", "include"), os.path.join(self.package_folder, "include"))
		
		copy(self, "*.a", self.build_folder, os.path.join(self.package_folder, "lib"), keep_path=False)
		copy(self, "*.lib", self.build_folder, os.path.join(self.package_folder, "lib"), keep_path=False)
		copy(self, "*.dll", self.build_folder, os.path.join(self.package_folder, "bin"), keep_path=False)
		copy(self, "*.so", self.build_folder, os.path.join(self.package_folder, "bin"), keep_path=False)
		copy(self, "*.dylib", self.build_folder, os.path.join(self.package_folder, "bin"), keep_path=False)

	def package_info(self):
		self.cpp_info.libs = collect_libs(self)
		
		if self.options.hot_reload:
			self.cpp_info.defines.append("HOT_RELOAD_ENABLED")
			
			if str(self.settings.compiler) in ["gcc", "clang"]:
				self.cpp_info.cxxflags.append("-fno-gnu-unique")

