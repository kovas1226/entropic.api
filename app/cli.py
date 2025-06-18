import argparse
from .api import FateOracle, quantum_seed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Query the Fate Oracle")
    parser.add_argument("--ask", metavar="QUESTION", help="Question for the oracle")
    args = parser.parse_args(argv)

    if args.ask:
        seed, _ = quantum_seed(args.ask)
        oracle = FateOracle(seed=seed)
        print(oracle.ask(args.ask))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
