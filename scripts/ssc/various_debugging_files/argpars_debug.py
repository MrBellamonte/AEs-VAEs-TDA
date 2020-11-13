import argparse


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', help='Print more data',
    action='store_false')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_input()

    if args.verbose:
        print('Stored true')

    else:
        print('.......ts.nothing')