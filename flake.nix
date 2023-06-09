{
  inputs = {
    # I typically use the exact nixpkgs set that I use for building my current
    # system to avoid redundancy.
    nixos-config.url = "github:dpaetzel/nixos-config";

    cmpbayes.url = "github:dpaetzel/cmpbayes/add-beta-binomial";
    cmpbayes.inputs.nixos-config.follows = "nixos-config";

    xcsf = {
      url = "github:dpaetzel/xcsf/flake";
      inputs.nixos-config.follows = "nixos-config";
    };
  };

  outputs = { self, nixos-config, cmpbayes, xcsf }:
    let
      nixpkgs = nixos-config.inputs.nixpkgs;
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
        '';

        postVenvCreation = ''
          unset SOURCE_DATE_EPOCH
          pip install pystan==3.4.0
        '';

      };
    };
}
