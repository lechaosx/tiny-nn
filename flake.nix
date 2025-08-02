{
	inputs = {
		nixpkgs.url = "github:NixOS/nixpkgs";
	};

	outputs = { nixpkgs, ... }:
	let
		pkgs = import nixpkgs { system = "x86_64-linux"; };
	in {
		devShells.x86_64-linux.default = pkgs.mkShell {
			buildInputs = with pkgs; [
				gcc
				gdb
				ninja
				cmake
				conan
				python3
				python3Packages.numpy
				python3Packages.torch
				python3Packages.torchvision
				godot
			];
		};
	};
}
