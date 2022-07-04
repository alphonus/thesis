with import <nixpkgs> {};
with pkgs.python3Packages;

let
  grinpy = buildPythonPackage rec {
    name = "grinpy";
    src = fetchFromGitHub {
      owner = "somacdivad";
      repo = "grinpy";
      rev = "597f910";
      sha256 = "1kq3j3vbr0xm8wg2jj4lri66g0v3bdsqzdpgpkdyl54mq4hnpxxv";
    };
  };

in pkgs.mkShell { buildInputs = [ numpy networkx grinpy ortools ]; }
