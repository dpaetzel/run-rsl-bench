{
  inputs = {
    nixpkgs.url = "github:dpaetzel/nixpkgs/dpaetzel/nixos-config";

    cmpbayes.url = "github:dpaetzel/cmpbayes/add-beta-binomial";
    cmpbayes.inputs.nixpkgs.follows = "nixpkgs";

    xcsf = {
      url = "github:dpaetzel/xcsf/flake";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, cmpbayes, xcsf }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python310;
    in rec {
      defaultPackage.${system} = python.pkgs.buildPythonPackage rec {
        pname = "TODO";
        version = "TODO";

        src = self;

        format = "pyproject";

        propagatedBuildInputs = with python.pkgs; [
          cmpbayes.defaultPackage."${system}"
          xcsf.defaultPackage."${system}"
          click
          matplotlib
          mlflow
          numpy
          pandas
          seaborn
          scipy
          scikit-learn
          toolz
          tqdm
        ];

        meta = with pkgs.lib; {
          description = "TODO";
          license = licenses.gpl3;
        };
      };

      devShell.${system} = pkgs.mkShell {

        buildInputs = with python.pkgs;
          [ ipython python venvShellHook pkgs.fish pkgs.parallel ]
          ++ defaultPackage.${system}.propagatedBuildInputs;

        venvDir = "./_venv";

        postShellHook = ''
          unset SOURCE_DATE_EPOCH

          export LD_LIBRARY_PATH="${
            pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]
          }:$LD_LIBRARY_PATH";

          export PYTHONPATH=src:$PYTHONPATH
        '';

        postVenvCreation = ''
          unset SOURCE_DATE_EPOCH
          pip install pystan==3.4.0
          pip install optuna
        '';
      };

      devShell.submit = pkgs.mkShell {
        buildInputs = [( python.withPackages(ps: [ps.click ps.mlflow]) )];

        # postShellHook is a Python thing (which is enabled, I think, by
        # venvShellHook?).
        shellHook = ''
          export PYTHONPATH=src:$PYTHONPATH
        '';
      };
    };
}
