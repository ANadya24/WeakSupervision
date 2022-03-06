import json
import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from segmentation.metric_evaluation.iou_calculation import calculate_iou
from segmentation.prediction.create_image_list import create_image_list

if __name__ == '__main__':

    with open(sys.argv[1], 'r') as file:
        config = json.load(file)

    res_df = pd.DataFrame(index=config['datasets'],
                          columns=config['mask_types'])
    os.makedirs(config["save_directory"], exist_ok=True)
    x = np.arange(0.1, 0.9, 0.05)
    for dataset in config['datasets']:

        os.makedirs(os.path.join(config["save_directory"], dataset), exist_ok=True)

        plt.figure(figsize=(10, 10))

        for mask_type in config['mask_types']:
            params_path = os.path.join(config["params_path"],
                                   dataset, 'iou_params.json')
            if not os.path.exists(params_path):
                params = None
            else:
                with open(params_path, 'r') as file:
                    params = json.load(file)

            image_pickle_path = os.path.join(config["image_path"], dataset,
                                             config['image_pickle_name'])
            if not os.path.exists(image_pickle_path):
                print('Didn\'t found image file, let\'s create it!')
                path_parts = image_pickle_path.split('/')
                create_image_list(dataset, config['init_image_path'], 'ST', '/'.join(path_parts[:-2]),
                                  num_seq=int(path_parts[-1].split('_')[0]))

            iou_cur = calculate_iou(image_pickle_path,
                                    os.path.join(config["masks_path"],
                                                 dataset, mask_type, config['mask_pickle_name']),
                                    params, margin=config["margin"])[0]

            plt.plot(x, iou_cur, label=f'{mask_type}_{max(iou_cur):.2f}')
            res_df.at[dataset, mask_type] = max(iou_cur)
        plt.legend()
        plt.title(f'{dataset}')
        plt.xlabel('Threshold')
        plt.ylabel('IOU')
        plt.grid()
        plt.savefig(os.path.join(config['save_directory'], dataset, 'iou.png'))

    res_df.to_csv(os.path.join(config['save_directory'], 'iou.csv'))
