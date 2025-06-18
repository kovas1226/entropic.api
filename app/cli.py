from .api import FateOracle


def main() -> None:
    oracle = FateOracle()
    while True:
        try:
            question = input("? ")
        except EOFError:
            break
        print(oracle.ask(question))


if __name__ == "__main__":
    main()
