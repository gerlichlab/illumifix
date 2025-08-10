{
  pkgs ? import (builtins.fetchGit {
    url = "https://github.com/NixOS/nixpkgs/";
    ref = "refs/tags/25.05";
  }) {}, 
  dev ? true,
  pipeline ? true,
}:
let py311 = pkgs.python311.withPackages (ps: with ps; [ numpy ]);
    poetryExtras = (
      [ ] ++ 
      (if dev then [ "coverage" "formatting" "linting" "pipeline" "testsuite" ] else [ ]) ++
      (if pipeline then [ "pipeline" ] else [ ])
    );
    poetryInstallExtras = (
      if poetryExtras == [] then ""
      else pkgs.lib.concatStrings [ " --with " (pkgs.lib.concatStringsSep "," poetryExtras) ]
    );
in
pkgs.mkShell {
  name = "illumifix-env";
  buildInputs = [ pkgs.poetry ];
  shellHook = ''
    # To get this working on the lab machine, we need to modify Poetry's keyring interaction:
    # https://stackoverflow.com/questions/74438817/poetry-failed-to-unlock-the-collection
    # https://github.com/python-poetry/poetry/issues/1917
    export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
    poetry env use "${py311}/bin/python"
    installcmd="poetry install -vvvv --sync${poetryInstallExtras}"
    echo "Running installation command: $installcmd"
    eval "$installcmd"
    source "$(poetry env info --path)/bin/activate"
  '';
}
