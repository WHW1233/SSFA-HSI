'''
This is the global settings of dataset paths.
'''
from pathlib import Path


# The root directory of all datasets
_root = (Path(__file__).parent / '/data/wanghuiwen').resolve()

known_datasets = {
    # Kodak images: http://r0k.us/graphics/kodak
    'kodak': _root / 'Kodak/kodak',

    # CLIC dataset: http://www.compression.cc
    'clic2022-test': _root / 'clic/test-2022',

    # Tecnick TESTIMAGES: https://testimages.org
    'tecnick-rgb-1200': _root / 'tecnick/TESTIMAGES/RGB/RGB_OR_1200x1200',

    # COCO dataset: http://cocodataset.org
    'coco-train2017': _root / 'train2017',
    'coco-val2017':   _root / 'val2017',

    # ImageNet dataset: http://www.image-net.org
    'imagenet-train': _root / 'imagenet/train',
    'imagenet-val':   _root / 'imagenet/val',

    # Vimeo-90k dataset: http://toflow.csail.mit.edu/
    'vimeo-90k': _root / 'vimeo-90k/sequences',

    # UVG dataset: http://ultravideo.fi/#testsequences
    'uvg-1080p': _root / 'video/uvg/1080p-frames',

    'hsiband3-train': _root / 'HSI/train/train_band3',
    'hsiband3-val':   _root / 'HSI/test/val_band3',

    'hsi-train': _root / 'HSI/train/train_hsi128',
    'hsi-val': _root / 'HSI/test/val_hsi128',

    'hysp11k-train': _root / 'HSI/train/train_hyspecnet11k',
    'hysp11k-val': _root / 'HSI/test/val_hyspecnet11k',

    'hysp11k-ext-train': '/media/nercms/EXTERNAL_USB/satellite_data/HSI_data/hyspecnet11k/data',
    'hysp11k-ext-val': '/media/nercms/EXTERNAL_USB/satellite_data/HSI_data/hyspecnet11k/test_data',

    'ohs-train': '/media/nercms/EXTERNAL_USB/satellite_data/WHU-OHS/OHS_MS_4_9_8bit_npy/train',
    'ohs-val': '/media/nercms/EXTERNAL_USB/satellite_data/WHU-OHS/OHS_MS_4_9_8bit_npy/test'


}
