import os
import pickle
from skimage import io
from skimage.morphology import dilation, square
from segmentation.draw_and_show.show_utils import plot_pair

def preprocess_label(mask):
    label = mask.copy()
    dilated = dilation(label, square(3))
    outline_mask = dilated != label
    label[outline_mask] = 0  # nicely creates 1-pixel border between touching cells
    label[label > 0] = 1  # change to binary
    return label


def create_image_list(dataset: str, data_path: str, seg_type: str = 'GT',
                      save: str = "", num_seq = 2, debug: bool = False, data_prefix=''):
    """
    Создать pickle файл с GT данными
    """
    path = f'{data_path}/{dataset}/0{num_seq}{data_prefix}/'
    label_path = f'{data_path}/{dataset}/0{num_seq}_GT/SEG/'
    if seg_type == 'GT':
        st_label_path = label_path
    else:
        st_label_path = f'{data_path}/{dataset}/0{num_seq}_ST/SEG/'

    full_fluo_paths = [path]
    fluo_label_paths = [label_path]
    img_label_pairs_fluo = []
    n_count_seq = {}
    for im_fluo_path, mask_fluo_path in zip(full_fluo_paths, fluo_label_paths):
        seq_tag = im_fluo_path.split('/')[2]
        count = 0
        for r, d, f in os.walk(mask_fluo_path):
            for file in f:
                if not file.endswith('.tif'):
                    continue
                if file.startswith('man_seg'):
                    t = file.split('.tif')[0].split('man_seg')[1]
                    mask_path = os.path.join(st_label_path, file)
                    t_full = int(t)
                    img_path = os.path.join(im_fluo_path, 't' + str(t_full).zfill(3) + '.tif')
                    img = io.imread(img_path)
                    label = io.imread(mask_path)
                    pair = (img, label)
                    print(img_path, mask_path)
                    img_label_pairs_fluo.append(pair)
                    count = count + 1
        n_count_seq[seq_tag] = count

    print('Number of fluo data: {}'.format(len(img_label_pairs_fluo)))

    if debug:
        # check pairs
        for img, mask in img_label_pairs_fluo:
            plot_pair(img / img.max(), mask > 0, 15)
            print('Image_shape is', img.shape, ', image_maximum is ', img.max())
            break

    if save != '':
        with open(f'{save}/{dataset}/0{num_seq}_{seg_type.lower()}_images.pkl', 'wb') as file:
            pickle.dump(img_label_pairs_fluo, file)

    return img_label_pairs_fluo