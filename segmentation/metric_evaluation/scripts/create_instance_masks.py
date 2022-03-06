import numpy as np
import os
import json
import pickle
import sys
from skimage import io

from segmentation.metric_evaluation.post_process import create_instance_masks


if __name__ == "__main__":
    config_path = sys.argv[1]
    with open(config_path, 'r') as file:
        config = json.load(file)

    mask_path = os.path.join(config["binary_masks_path"], config["dataset"], config["type_seg"].replace('01', '02'),
                             config["mask_filename"])
    with open(mask_path, 'rb') as file:
        res_masks = pickle.load(file)

    optuna_path = os.path.join(config["parameters_path"], config["dataset"], config["type_seg"], "best_params.json")
    with open(optuna_path, 'r') as file:
        optuna_params = json.load(file)

    instance_masks = create_instance_masks(res_masks, optuna_params)

    os.makedirs(config["output_path"], exist_ok=True)
    os.makedirs(os.path.join(config["output_path"], config["dataset"]), exist_ok=True)
    out_path = os.path.join(config["output_path"], config["dataset"],
                config["type_seg"].replace('01', '02'))
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, config["mask_filename"]), 'wb') as file:
        pickle.dump(instance_masks, file)

    for i, mask in enumerate(instance_masks):
        io.imsave(out_path + f'/{i}.png', np.uint16(mask))
