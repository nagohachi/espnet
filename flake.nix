{
  description = "Python Shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            pkgs.python312
            pkgs.uv
            pkgs.sox
            pkgs.ffmpeg
            pkgs.cudaPackages_12_8.cudatoolkit
            pkgs.cudaPackages_12_8.cudnn
          ];
          shellHook = ''
            export PS1="(nix) $PS1"
            echo "uv version: $(uv --version)"
            echo "python version: $(python --version)"
            echo "CUDA version: $(nvcc --version | grep release)"
            echo "Building environment with uv......"
            uv sync
          '';
        };
      });
}
