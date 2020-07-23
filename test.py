import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                            help='node rank for distributed training')
args = parser.parse_args()
#print(args.rank)
print(args.local_rank)






