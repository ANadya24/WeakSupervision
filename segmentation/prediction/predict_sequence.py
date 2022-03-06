import segmentation_models_pytorch as sm
import pickle
import json
import sys
import os

from predict_utils import (
    load_checkpoint,
    load_checkpoint_w_model,
    predict_and_save,
    predict_and_plot
)
from create_image_list import create_image_list

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as file:
        config = json.load(file)
    
    checkpoint_path = os.path.join(config['checkpoint_path'], config['dataset'], config['seg_type'], 'checkpoints', config['checkpoint_name'])
    if not config['checkpoint_w_model']:
        model = sm.Unet(config['backbone'], classes=2, activation='sigmoid')
        model = load_checkpoint(model, checkpoint_path)
    else:
        model = load_checkpoint_w_model(checkpoint_path)

    model = model.to(config["device"])
    
    if not os.path.exists(config['image_pickle_path']):
        print('Didn\'t found image file, let\'s create it!')
        path_parts = config['image_pickle_path'].split('/')
        if config['image_pickle_path'].find('_gt') > 0:
            seg_type = 'GT'
        else:
            seg_type = 'ST'
        create_image_list(config['dataset'], config['image_path'],  seg_type,
                          '/'.join(path_parts[:-2]), num_seq=int(path_parts[-1].split('_')[0]),
                          data_prefix=config["data_prefix"])
    with open(config['image_pickle_path'], 'rb') as file:
        img_label_pairs_fluo = pickle.load(file)
        
#     from skimage.exposure import equalize_adapthist
#     img_label_pairs_fluo = [(equalize_adapthist(img, clip_limit=0.03), mask) for (img, mask) in img_label_pairs_fluo]

    num_seq = int(config['image_pickle_path'].split('/')[-1].split('_')[0])

    out_dir = os.path.join(config["out_path"], config["dataset"], config["seg_type"], str(num_seq).zfill(2), config["mask_pickle_name"])
    os.makedirs(os.path.join(config["out_path"], config["dataset"], config["seg_type"], str(num_seq).zfill(2)), exist_ok=True)

    if config["test"]:
        _ = predict_and_plot(img_label_pairs_fluo, model, num_frames=2, device=config["device"])
        _ = predict_and_save(img_label_pairs_fluo, model, out_dir, num_frames=2, device=config["device"],
                             adjust_gamma=config['adjust_gamma'], gamma=config['gamma'])
    else:
        _ = predict_and_save(img_label_pairs_fluo, model, out_dir, device=config["device"],
                             adjust_gamma=config['adjust_gamma'], gamma=config['gamma'])


