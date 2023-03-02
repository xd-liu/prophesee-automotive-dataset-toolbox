import argparse
import os
import numpy as np
from src.io.psee_loader import PSEELoader
from src.io.box_loading import reformat_boxes

def main(args):
    '''read label file'''
    root_dir = args.root_dir
    file_list = os.listdir(root_dir)
    for file in file_list:
        if ".npy" not in file:
            continue

        
        label = np.load(os.path.join(root_dir, file), allow_pickle=True)
        label = reformat_boxes(label)
        # print(label.dtype.names)
        t = label['t'].astype(np.float)
        uni_t = np.unique(t)
        diff_t = np.diff(uni_t)
        remainder = np.mod(diff_t, 500000)
        # print(remainder)
        strange = diff_t[[remainder > 0]]
        if len(strange) > 0:
            print("Processing file: {}".format(file))
            print(strange[0])


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