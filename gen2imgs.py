'''Convert GEN1 event data to images.'''
import argparse
import os
import numpy as np
from src.io.psee_loader import PSEELoader
import cv2

SENSOR_SIZE = (240, 304)
LABEL_RATE = 500000


def make_event_histogram(args, events, shape=(240, 304)):
    """Event polarity histogram with shape [H, W, 2]."""
    H, W = shape

    # count the number of positive and negative events per pixel
    pos = events[events[:, 3] > 0]
    neg = events[events[:, 3] < 0]

    pos_x, pos_y = pos[:, 0].astype(np.int32), pos[:, 1].astype(np.int32)
    pos_count = np.bincount(pos_x + pos_y * W, minlength=H * W).reshape(H, W)
    neg_x, neg_y = neg[:, 0].astype(np.int32), neg[:, 1].astype(np.int32)
    neg_count = np.bincount(neg_x + neg_y * W, minlength=H * W).reshape(H, W)

    # [H, W, 2]
    result = np.stack([pos_count, neg_count], axis=2)
    return result


def vis_hist(args, hist_nd):
    img = np.stack([
        hist_nd[:, :, 0],
        np.zeros(hist_nd[:, :, 0].shape),
        hist_nd[:, :, 1],
    ],
                   axis=2)

    mean = np.mean(img)
    std = np.std(img)

    img[img > (mean + 10 * std)] = 0.

    max_value = np.max(img)
    min_value = np.min(img)

    if max_value == min_value:
        if max_value > 0:
            img = img / max_value
    else:
        img = (img - min_value) / (max_value - min_value)

    mask = np.where(np.sum(img, 2) == 0)
    img[mask] = [1., 1., 1.]

    # weight_mask = np.sum(img, 2)
    # max_weight = 2.
    # weight_mask[weight_mask > max_weight] = max_weight
    # weight_mask = weight_mask / weight_mask.max()
    background = np.ones_like(img)
    # dev
    weight_mask = np.ones_like(np.sum(img, 2))
    img = weight_mask[...,
                      None] * img + (1. - weight_mask[..., None]) * background
    img = img * 255.

    img = img.astype('uint8')
    if args.bw:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def event2frame(args, events):
    events = np.vstack([
        events['x'], events['y'], events['t'],
        events['p'].astype(np.float) * 2 - 1
    ]).T

    events = events.astype(np.float)
    # print(events[:,3].min(), )s
    hist_nd = make_event_histogram(args, events, shape=SENSOR_SIZE)
    img = vis_hist(args, hist_nd)
    return img


def main(args):

    root_dir = args.root_dir
    output_dir = args.output_dir + f"_delta_{args.delta_t}_min_{args.min_event_num}"
    if args.bw:
        output_dir += "_bw"

    delta_t = args.delta_t

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    event_files = os.listdir(root_dir)
    for event_file in event_files:
        if ".npy" in event_file:
            continue

        print("Processing file: {}".format(event_file))
        subdir = os.path.join(output_dir, event_file.split(".")[0])
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        video = PSEELoader(os.path.join(root_dir, event_file))
        drop_start = video.load_delta_t(LABEL_RATE + delta_t / 2)

        # to iterate through a video:
        while not video.done:
            # load events and boxes from all files
            events = video.load_delta_t(delta_t)
            # do something with the events

            event_num = len(events)
            if event_num < args.min_event_num:
                continue

            curr_time = events[-1]['t']
            img_name = f"{curr_time}_{event_num}.png"
            img = event2frame(args, events)
            cv2.imwrite(os.path.join(subdir, img_name), img)
            if LABEL_RATE - delta_t > 0:
                drop_intervel = video.load_delta_t(LABEL_RATE - delta_t)

        if args.dev:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default=
        "/home/xudong99/scratch/GEN1/detection_dataset_duration_60s_ratio_1.0/test",
        help="Path to the root directory of the dataset.")
    parser.add_argument("--output_dir",
                        type=str,
                        default="/home/xudong99/scratch/GEN1/test_imgs",
                        help="Path to the output directory.")
    parser.add_argument("--delta_t",
                        type=int,
                        default=500000,
                        help="The time interval between two frames.")
    parser.add_argument("--min_event_num",
                        type=int,
                        default=1000,
                        help="The minimum number of events in a frame.")
    parser.add_argument("--dev",
                        action="store_true",
                        help="Whether to run in dev mode.")
    parser.add_argument(
        "--bw",
        action="store_true",
        help="Whether to convert the output to black and white.")
    args = parser.parse_args()
    main(args)
