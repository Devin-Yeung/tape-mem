{
  pkgs,
  lib,
  ...
}:
{
  git-hooks.hooks.markdown-fmt = {
    enable = true;
    name = "format markdown files";
    entry = "${lib.getExe pkgs.oxfmt}";
    files = "\\.md$";
    excludes = [
      ".devenv"
    ];
    language = "system";
    pass_filenames = true; # in case that oxfmt touch other supported files
  };
}
