{
	description = "C++ Development Environment";

	inputs = {
		nixpkgs.url = "github:NixOS/nixpkgs";
	};

	outputs = { nixpkgs, ... }:
	let
		pkgs = import nixpkgs { system = "x86_64-linux"; };
	in {
		devShells.x86_64-linux.default = pkgs.mkShell {
			name = "cpp-dev-env";

			buildInputs = [
				pkgs.gcc
				pkgs.gdb
				pkgs.ninja
				pkgs.cmake
				pkgs.conan
			];
		};
	};
}
