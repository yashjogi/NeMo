import sys
sys.path = ["/home/lab/NeMo"] + sys.path

import argparse
from pathlib import Path

import torch

from nemo.collections.nlp.models import PunctuationCapitalizationModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "-c", type=Path, help="Path to checkpoint to encapsulate.", required=True)
    parser.add_argument("--nemo", "-n", type=Path, help="Path to output nemo file", required=True)
    parser.add_argument("--cfg", "-f", type=Path, help="Path to config file", required=True)
    args = parser.parse_args()
    args.ckpt = args.ckpt.expanduser()
    args.nemo = args.nemo.expanduser()
    args.cfg = args.cfg.expanduser()
    return args


def main():
    args = get_args()
    model = PunctuationCapitalizationModel(args.cfg.model)
    model.load_state_dict(torch.load(args.ckpt))
    model.save_to(args.nemo)


if __name__ == "__main__":
    main()
