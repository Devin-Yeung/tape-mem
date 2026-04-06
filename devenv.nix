{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  env.HF_HOME = "${config.devenv.state}/hf";

  imports = [
    ./devenv/astral
    ./devenv/jupyter-notebook.nix
    ./devenv/markdown-fmt.nix
  ];

  # https://devenv.sh/packages/
  packages = [ ];

  # https://devenv.sh/languages/
  languages = {
    nix.enable = true;
  };

  # https://devenv.sh/basics/
  enterShell = ''
    python3 --version
  '';

  # https://devenv.sh/git-hooks/
  git-hooks.hooks = {
    nixfmt.enable = true;
    taplo.enable = true;
  };

  # astral toolchain
  astral = {
    lint.enable = true;
    format.enable = true;
    ty.enable = true;
    uv-export.enable = true;
  };

  # See full reference at https://devenv.sh/reference/options/
}
