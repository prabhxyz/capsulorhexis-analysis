import sys
import argparse

# Optional hack to ensure local packages are found:
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the "main" functions from the separate training scripts
from training.train_phase import main_phase_train
from training.train_segmentation import main_seg_train

def main():
    parser = argparse.ArgumentParser(description="Train both phase & segmentation models.")
    parser.add_argument("--phase_root", type=str, required=True,
                        help="Path to the Cataract-1k-Phase dataset.")
    parser.add_argument("--phase_epochs", type=int, default=10)
    parser.add_argument("--phase_batch", type=int, default=8)
    parser.add_argument("--phase_lr", type=float, default=1e-4)

    parser.add_argument("--seg_root", type=str, required=True,
                        help="Path to the Cataract-1k-Seg dataset.")
    parser.add_argument("--seg_epochs", type=int, default=10)
    parser.add_argument("--seg_batch", type=int, default=4)
    parser.add_argument("--seg_lr", type=float, default=1e-4)

    args = parser.parse_args()

    print("=== TRAINING PHASE MODEL ===")
    main_phase_train(
        root_dir=args.phase_root,
        epochs=args.phase_epochs,
        batch_size=args.phase_batch,
        lr=args.phase_lr
    )

    print("=== TRAINING SEGMENTATION MODEL ===")
    main_seg_train(
        root_dir=args.seg_root,
        epochs=args.seg_epochs,
        batch_size=args.seg_batch,
        lr=args.seg_lr
    )


if __name__ == "__main__":
    main()