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
				emscripten
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
