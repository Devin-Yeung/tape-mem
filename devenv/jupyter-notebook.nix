{
  pkgs,
  ...
}:
{
  git-hooks.hooks.nbconvert = {
    enable = true;
    # The name of the hook (appears on the report table):
    name = "Clear Jupyter Notebook Output and Metadata";
    # The command to execute (mandatory):
    entry = "${pkgs.python313Packages.nbconvert}/bin/jupyter-nbconvert --clear-output --inplace --ClearMetadataPreprocessor.enabled=True";
    # The pattern of files to run on (default: "" (all))
    # see also https://pre-commit.com/#hooks-files
    files = "\\.ipynb$";
    # Exclude files that were matched by these patterns (default: [ ] (none)):
    excludes = [
      ".devenv"
      ".venv"
    ];
    # The language of the hook - tells pre-commit
    # how to install the hook (default: "system")
    # see also https://pre-commit.com/#supported-languages
    language = "system";
    # Set this to false to not pass the changed files
    # to the command (default: true):
    pass_filenames = true;
  };
}
