{
  description = "TensorC Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
              clang
              clang-tools
          ];

          shellHook = ''
            echo "================================"
            echo " TensorC Development"
            echo "================================"
            echo
          '';
        };
      });
}
