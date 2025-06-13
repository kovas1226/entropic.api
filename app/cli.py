import argparse
import math

from .api import (
    example_bell_and_qft,
    grover_search,
    teleport,
    ghz_circuit,
    visualize_probabilities,
    run_demo,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Command line utilities for the symbolic quantum API"
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("bell", help="Run Bell + QFT example")
    sub.add_parser("grover", help="Grover search demo")
    sub.add_parser("teleport", help="Teleport |+> state")
    sub.add_parser("ghz", help="Prepare GHZ state")
    sub.add_parser("demo", help="Run combined demo")
    args = parser.parse_args(argv)

    if args.command == "bell":
        results, state = example_bell_and_qft()
        print("Bell measurement:", results)
        visualize_probabilities(state)
    elif args.command == "grover":
        sol, state = grover_search([3], 2, iterations=1)
        print("Grover result bits:", sol)
        visualize_probabilities(state)
    elif args.command == "teleport":
        m, final = teleport([1 / math.sqrt(2), 1 / math.sqrt(2)])
        print("Bell outcomes:", m)
        visualize_probabilities(final)
    elif args.command == "ghz":
        circ = ghz_circuit(3)
        visualize_probabilities(circ.state)
    else:
        run_demo()


if __name__ == "__main__":
    main()