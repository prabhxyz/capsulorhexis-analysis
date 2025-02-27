import argparse
import os
import torch
import torch.backends.cudnn as cudnn

from .train_phase import main as train_phase_main
from .train_segmentation import main as train_seg_main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase_root", type=str, default=None)
    parser.add_argument("--seg_root", type=str, default=None)
    parser.add_argument("--phase_epochs", type=int, default=10)
    parser.add_argument("--seg_epochs", type=int, default=10)
    parser.add_argument("--batch_size_phase", type=int, default=8)
    parser.add_argument("--batch_size_seg", type=int, default=6)
    parser.add_argument("--lr_phase", type=float, default=1e-4)
    parser.add_argument("--lr_seg", type=float, default=1e-4)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    # Enable benchmark
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    cudnn.benchmark = True

    # 1) Train Phase Model if path given
    if args.phase_root is not None:
        phase_args = [
            "--root_dir", args.phase_root,
            "--phase_epochs", str(args.phase_epochs),
            "--batch_size", str(args.batch_size_phase),
            "--lr", str(args.lr_phase),
        ]
        if args.no_cuda:
            phase_args.append("--no_cuda")
        print("=== TRAINING PHASE MODEL ===")
        train_phase_main(phase_args)
    else:
        print("No phase_root specified => skipping phase training.")

    # 2) Train Segmentation Model if path given
    if args.seg_root is not None:
        seg_args = [
            "--root_dir", args.seg_root,
            "--seg_epochs", str(args.seg_epochs),
            "--batch_size", str(args.batch_size_seg),
            "--lr", str(args.lr_seg),
        ]
        if args.no_cuda:
            seg_args.append("--no_cuda")
        print("=== TRAINING SEGMENTATION MODEL ===")
        train_seg_main(seg_args)
    else:
        print("No seg_root specified => skipping segmentation.")

if __name__ == "__main__":
    main()
