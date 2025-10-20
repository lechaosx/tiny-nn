{
	inputs = {
		nixpkgs.url = "github:NixOS/nixpkgs";
	};

	outputs = { nixpkgs, ... }:
	let
		pkgs = import nixpkgs { system = "x86_64-linux"; };
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
				pkgs.emscripten
			];

			shellHook = ''
				REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
				export EM_CACHE="$REPO_ROOT/.emscripten_cache"

				if [ ! -d "$EM_CACHE" ]; then
					cp -r ${pkgs.emscripten}/share/emscripten/cache "$EM_CACHE"
					chmod u+rwX -R "$EM_CACHE"
				fi
			'';
		};
	};
}
