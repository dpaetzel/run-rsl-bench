{
  inputs = {
    nixpkgs.url = "github:dpaetzel/nixpkgs/dpaetzel/nixos-config";

    cmpbayes.url = "github:dpaetzel/cmpbayes/add-beta-binomial";
    cmpbayes.inputs.nixpkgs.follows = "nixpkgs";

    mlflowExportImportSrc = {
      url = "github:mlflow/mlflow-export-import/f9bba63";
      flake = false;
    };

    suprb = {
      url = "github:dpaetzel/suprb/make-flake";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    xcsf = {
      url = "github:dpaetzel/xcsf/flake";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    linearTreeSrc = {
      url = "github:cerlymarco/linear-tree";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, cmpbayes, mlflowExportImportSrc, suprb, xcsf, linearTreeSrc }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python310;

      mlflow-import-export = python.pkgs.buildPythonApplication rec {
        pname = "mlflow-import-export";
        version = "dev";
        src = mlflowExportImportSrc;

        postPatch = ''
          sed -i 's/pandas>=1.5.2/pandas/' setup.py
          sed -i 's/mlflow-skinny>=2.2.2/mlflow/' setup.py
        '';

        doCheck = false;

        propagatedBuildInputs = with python.pkgs; [ mlflow pandas ];
      };

      linear-tree = python.pkgs.buildPythonPackage rec {
        pname = "linear-tree";
        version = "dev";
        src = linearTreeSrc;

        propagatedBuildInputs = with python.pkgs; [ scipy numpy scikit-learn ];

      };
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
          optuna
          pandas
          seaborn
          scipy
          scikit-learn
          suprb.packages."${system}".default
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
          # pip install optuna
        '';
      };

      devShell.submit = pkgs.mkShell {
        buildInputs = [ (python.withPackages (ps: [ ps.click ps.mlflow ps.setuptools ])) ];

        # postShellHook is a Python thing (which is enabled, I think, by
        # venvShellHook?).
        shellHook = ''
          export PYTHONPATH=src:$PYTHONPATH
        '';
      };
    };
}
