{
	inputs = {
		nixpkgs.url            = "github:NixOS/nixpkgs";
		nixpkgs-emscripten.url = "github:NixOS/nixpkgs/nixos-24.11";
	};

	outputs = { nixpkgs, nixpkgs-emscripten, ... }:
	let
		pkgs            = import nixpkgs            { system = "x86_64-linux"; };
		pkgs-emscripten = import nixpkgs-emscripten { system = "x86_64-linux"; };
	in {
		devShells.x86_64-linux.default = pkgs.mkShell {
			buildInputs = [
				pkgs.gcc
				pkgs.gdb
				pkgs.ninja
				pkgs.cmake
				pkgs.conan
				pkgs.python3
				pkgs.python3Packages.numpy
				pkgs.python3Packages.torch
				pkgs.python3Packages.torchvision
				pkgs.godot
				pkgs-emscripten.emscripten
			];

			shellHook = ''
				REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
				export EM_CACHE="$REPO_ROOT/.emscripten_cache"

				if [ ! -d "$EM_CACHE" ]; then
					cp -r ${pkgs-emscripten.emscripten}/share/emscripten/cache "$EM_CACHE"
					chmod u+rwX -R "$EM_CACHE"
				fi
			'';
		};
	};
}
