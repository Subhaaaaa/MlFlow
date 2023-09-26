import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--name','-n', default='Subha', type=str)
    args.add_argument('--age', '-a', default=23.0, type=float)
    parse_args=args.parse_args()

    print(parse_args.name, parse_args.age)

    #Now we can apss these arguments from command line itself : test.py --name "Subhu" --age 23.0





