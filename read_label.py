import argparse
import os
import numpy as np
from src.io.psee_loader import PSEELoader


def main(args):
    '''read label file'''
    root_dir = args.root_dir
    file_list = os.listdir(root_dir)
    for file in file_list:
        if ".npy" not in file:
            continue

        print("Processing file: {}".format(file))
        label = np.load(os.path.join(root_dir, file), allow_pickle=True)
        t = label['t'].astype(np.float)
        uni_t = np.unique(t)
        diff_t = np.diff(uni_t)
        print(diff_t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_dir',
        type=str,
        default=
        '/home/xudong99/scratch/GEN1/detection_dataset_duration_60s_ratio_1.0/test',
        help='Path to the root directory of the dataset.')
    args = parser.parse_args()
    main(args)