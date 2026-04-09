import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + "/../.."))
from basicts import launch_training


def parse_args():
    parser = ArgumentParser(description="ST-UFNet: Spatio-Temporal U-shaped Flow Matching Network")
    parser.add_argument("-c", "--cfg", default="ST_MFNet_PEMS08.py", help="training config")
    parser.add_argument("--gpus", default="0", help="visible gpus")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    launch_training(args.cfg, args.gpus)
