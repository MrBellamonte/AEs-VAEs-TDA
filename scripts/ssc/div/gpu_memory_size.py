import argparse

import torch

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--path", type=str)
    parser.add_argument('-bs', "--batch_size",
                        default=64,
                        help="number of parallel processes", type=int)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_input()

    dataset = torch.load('{}/full_dataset.pt'.format(args.path))

    x = dataset[:args.batch_size][0].to('cuda')
    print(torch.cuda.memory_allocated())
