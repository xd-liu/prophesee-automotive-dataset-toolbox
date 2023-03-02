import numpy as np
import os
import json
import argparse

BBOX_DTYPE = np.dtype({
    'names':
    ['t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence'],
    'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<f4'],
    'offsets': [0, 8, 12, 16, 20, 24, 28, 32],
    'itemsize':
    40
})

CAR_CLSSSES = [
    "car", "truck", "bus", "trailer", "van", "boat", "motorcycle", "bicycle",
    "train", "airplane"
]
PERDESTRIAN_CLASSES = ["pedestrian", "rider", "person"]


def main(args):
    fn = os.path.join(args.input_dir, "output.json")
    with open(fn, 'r') as f:
        data = json.load(f)

    valid_list = []
    for item in data:
        time = int(item['file_name'].split('_')[0])
        for bbox, label, score in zip(item["boxex"], item["labels"],
                                      item["scores"]):
            for cls in CAR_CLSSSES:
                if cls in label:
                    valid_list.append(
                        [time, bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, score])
                    break

            for cls in PERDESTRIAN_CLASSES:
                if cls in label:
                    valid_list.append(
                        [time, bbox[0], bbox[1], bbox[2], bbox[3], 1, 0, score])
                    break

    valid_list = np.array(valid_list, dtype=BBOX_DTYPE)
    res = np.zeros((len(valid_list)), dtype=BBOX_DTYPE)
    res['t'] = valid_list['t']
    res['x'] = (valid_list['x'] + valid_list['w']) / 2
    res['y'] = (valid_list['y'] + valid_list['h']) / 2
    res['w'] = valid_list['w'] - valid_list['x']
    res['h'] = valid_list['h'] - valid_list['y']
    res['class_id'] = valid_list['class_id']
    res['track_id'] = valid_list['track_id']
    res['class_confidence'] = valid_list['class_confidence']

    if args.output_dir == "":
        args.output_dir = os.path.split(args.input_dir)[-1]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    np.save(os.path.join(args.output_dir, "prediction.npy"), res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=
        "/home/xudong99/scratch/GEN1/Detic/detic_output/hist_500000_min_1000_voc_coco_score_0.1"
    )
    parser.add_argument("--output_dir", type=str, default="")

    args = parser.parse_args()
    main(args)