import json
import sys
import pickle
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from segmentation.metric_evaluation.seg_calculation import calculate_seg

if __name__ == '__main__':

    with open(sys.argv[1], 'r') as file:
        config = json.load(file)

    res_df = pd.DataFrame(index=config['datasets'],
                          columns=config['mask_types'])
    os.makedirs(config["save_directory"], exist_ok=True)

    if not config['calc_for_best_thr']:
        if not config['thrs']:
            thrs = np.arange(0.1, 0.9, 0.05)
        else:
            thrs = config['thrs']

        if isinstance(thrs, int):
            thrs = [thrs]

    for dataset in config['datasets']:

        os.makedirs(os.path.join(config["save_directory"], dataset), exist_ok=True)

        plt.figure(figsize=(10, 10))

        for mask_type in config['mask_types']:
            with open(os.path.join(config["params_path"],
                                   dataset, 'best_params.json'), 'r') as file:
                params = json.load(file)

            if not config['calc_for_best_thr']:
                params.pop('thr')
            else:
                thrs = params.pop('thr')
            seg_cur = calculate_seg(os.path.join(config["image_path"], dataset, config['image_pickle_name']),
                                    os.path.join(config["masks_path"], dataset, config['mask_pickle_name']),
                                    params, thrs=thrs)[0]
            if isinstance(seg_cur, float):
                seg_cur = [seg_cur]

            plt.plot(thrs, seg_cur, label=f'{mask_type}_{max(seg_cur):.2f}')
            res_df.at[dataset, mask_type] = max(seg_cur)
        plt.legend()
        plt.title(f'{dataset}')
        plt.xlabel('Threshold')
        plt.ylabel('SEG')
        plt.grid()
        plt.savefig(os.path.join(config['save_directory'], dataset, 'seg.png'))
    res_df.to_csv(os.path.join(config['save_directory'], 'seg.csv'))