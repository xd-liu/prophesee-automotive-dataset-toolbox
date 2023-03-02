import os
import numpy as np
import cv2
from detic import detic_predict
from detectron2.utils.visualizer import Visualizer
import json


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    exp_dir = os.path.join(output_dir, args.exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    img_list = os.listdir(input_dir)
    out_put_list = []
    for img_name in img_list:
        if ".png" not in img_name:
            continue

        print("Processing file: {}".format(img_name))
        im = cv2.imread(os.path.join(input_dir, img_name))
        metadata, outputs = detic_predict(im)
        if args.vis:
            v = Visualizer(im[:, :, ::-1], metadata)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(os.path.join(exp_dir, img_name),
                        out.get_image()[:, :, ::-1])

        out_put_list.append({
            "file_name":
            img_name,
            "boxes":
            outputs["instances"].pred_boxes,
            "scores":
            outputs["instances"].scores,
            "labels": [
                metadata.thing_classes[x]
                for x in outputs["instances"].pred_classes.cpu().tolist()
            ]
        })
        if args.dev:
            with open(os.path.join(exp_dir, "output_dev.json"), "w") as f:
                json.dump(out_put_list, f)
            break

    with open(os.path.join(exp_dir, "output.json"), "w") as f:
        json.dump(out_put_list, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=
        "/home/xudong99/scratch/GEN1/test_imgs_delta_500000_min_1000/17-10-06_13-18-33_732500000_792500000_td"
    )
    parser.add_argument("--output_dir", type=str, default="./detic_output/")
    parser.add_argument("--exp_name", type=str, default="hist_500000_min_1000")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    main(args)