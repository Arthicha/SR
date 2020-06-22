import os
import sys
print(__file__)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
import importlib
from utils.solver import Solver
#python carn/train.py --patch_size 64 --batch_size 20 --max_steps 600000 --decay 400000 --model carn --ckpt_name carn --ckpt_dir checkpoint/carn --scale 0 --num_gpu 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="refiner")
    parser.add_argument("--ckpt_name", type=str, default='checkpoint/checkpoint.pth')
    parser.add_argument("--saved_ckpt_dir", type=str, default='checkpoint/new')
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--train_data_path", type=str,default="dataset/DIV2K_train.h5")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--verbose", action="store_true", default="store_true")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--decay", type=int, default=400000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=10.0)
    return parser.parse_args()

def main(cfg):
    # dynamic import using --model argument
    net = importlib.import_module("model.{}".format(cfg.model)).Net
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    
    solver = Solver(net, cfg)
    solver.fit()

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
