from os.path import join
import random
import numpy as np
from scipy.io import loadmat
import torch.utils.data as data
import util.util_img


class Dataset(data.Dataset):
    data_root = '/media/Data/dsl-course/ycb_genre2'
    list_root = join(data_root, 'status')
    status_and_suffix = {
        'rgb': {
            'status': 'rgb.txt',
            'suffix': '_rgb.png',
        },
        'depth': {
            'status': 'depth.txt',
            'suffix': '_depth.png',
        },
        'depth_minmax': {
            'status': 'depth_minmax.txt',
            'suffix': '_minmax.npy',
        },
        'trans_mat': {
            'status': 'trans_mat.txt',
            'suffix': '_trans_mat.npy',
        },
        'silhou': {
            'status': 'silhou.txt',
            'suffix': '_silhouette.png',
        },
        'normal': {
            'status': 'normal.txt',
            'suffix': '_normal.png'
        },
        'voxel': {
            'status': 'vox_rot.txt',
            'suffix': '_voxel_rot.npz'
        },
        'spherical': {
            'status': 'spherical.txt',
            'suffix': '_spherical.npz'
        },
        'voxel_canon': {
            'status': 'vox_canon.txt',
            'suffix': '_voxel_normalized_128.mat'
        },
    }
    class_aliases = {
        'mastercan':  '002_master_chef_can',
        'crackerbox': '003_cracker_box',
        'sugarbox':   '004_sugar_box',
        'tomatocan':  '005_tomato_soup_can',
        'mustard':    '006_mustard_bottle',
        'tunacan':    '007_tuna_fish_can',
        'pudbox':     '008_pudding_box',
        'gelbox':     '009_gelatin_box',
        'meatcan':    '010_potted_meat_can',
        'banana':     '011_banana',
        'pitcher':    '019_pitcher_base',
        'bleach':     '021_bleach_cleanser',
        'bowl':       '024_bowl',
        'mug':        '025_mug',
        'drill':      '035_power_drill',
        'block':      '036_wood_block',
        'scissors':   '037_scissors',
        'marker':     '040_large_marker',
        'lclamp':     '051_large_clamp',
        'elclamp':    '052_extra_large_clamp',
        'brick':      '061_foam_brick',
        'all':        '002_master_chef_can+003_cracker_box+004_sugar_box+005_tomato_soup_can+006_mustard_bottle+007_tuna_fish_can+008_pudding_box+009_gelatin_box+010_potted_meat_can+011_banana+019_pitcher_base+021_bleach_cleanser+024_bowl+025_mug+035_power_drill+036_wood_block+037_scissors+040_large_marker+051_large_clamp+052_extra_large_clamp+061_foam_brick',
    }
    class_list = class_aliases['all'].split('+')

    @classmethod
    def add_arguments(cls, parser):
        return parser, set()

    @classmethod
    def read_bool_status(cls, status_file):
        with open(join(cls.list_root, status_file)) as f:
            lines = f.read()
        return [x == 'True' for x in lines.split('\n')[:-1]]

    def __init__(self, opt, mode='train', model=None):
        assert mode in ('train', 'vali')
        self.mode = mode
        if model is None:
            required = ['rgb']
            self.preproc = None
        else:
            required = model.requires
            self.preproc = model.preprocess

        # Parse classes
        classes = []  # alias to real for locating data
        class_str = ''  # real to alias for logging
        for c in opt.classes.split('+'):
            class_str += c + '+'
            if c in self.class_aliases:  # nickname given
                classes += self.class_aliases[c].split('+')
            else:
                classes = c.split('+')
        class_str = class_str[:-1]  # removes the final +
        classes = sorted(list(set(classes)))

        # Load items and train-test split
        with open(join(self.list_root, 'items_all.txt')) as f:
            lines = f.read()
        item_list = lines.split('\n')[:-1]
        is_train = self.read_bool_status('is_train.txt')
        assert len(item_list) == len(is_train)

        # Load status the network requires
        has = {}
        for data_type in required:
            assert data_type in self.status_and_suffix.keys(), \
                "%s required, but unspecified in status_and_suffix" % data_type
            has[data_type] = self.read_bool_status(
                self.status_and_suffix[data_type]['status']
            )
            assert len(has[data_type]) == len(item_list)

        # Pack paths into a dict
        samples = []
        for i, item in enumerate(item_list):
            class_id, _ = item.split('/')[:2] # Check this line!
            item_in_split = ((self.mode == 'train') == is_train[i])
            if item_in_split and class_id in classes:
                # Look up subclass_id for this item
                sample_dict = {'item': join(self.data_root, item)}
                # As long as a type is required, it appears as a key
                # If it doens't exist, its value will be None
                for data_type in required:
                    suffix = self.status_and_suffix[data_type]['suffix']
                    k = data_type + '_path'
                    if data_type == 'voxel_canon':
                        # All different views share the same canonical voxel
                        sample_dict[k] = join(self.data_root, item.split('_view')[0] + suffix) \
                            if has[data_type][i] else None
                    else:
                        sample_dict[k] = join(self.data_root, item + suffix) \
                            if has[data_type][i] else None
                if None not in sample_dict.values():
                    # All that are required exist
                    samples.append(sample_dict)

        # If validation, dataloader shuffle will be off, so need to DETERMINISTICALLY
        # shuffle here to have a bit of every class
        if self.mode == 'vali':
            if opt.manual_seed:
                seed = opt.manual_seed
            else:
                seed = 0
            random.Random(seed).shuffle(samples)
        self.samples = samples

    def __getitem__(self, i):
        sample_loaded = {}
        for k, v in self.samples[i].items():
            sample_loaded[k] = v  # as-is
            if k.endswith('_path'):
                if v.endswith('.png'):
                    im = util.util_img.imread_wrapper(
                        v, util.util_img.IMREAD_UNCHANGED,
                        output_channel_order='RGB')
                    # Normalize to [0, 1] floats
                    im = im.astype(float) / float(np.iinfo(im.dtype).max)
                    sample_loaded[k[:-5]] = im
                elif v.endswith('_minmax.npy'):
                    sample_loaded['depth_minmax'] = np.load(v)
                elif v.endswith('_trans_mat.npy'):
                    sample_loaded['trans_mat'] = np.load(v)
                elif v.endswith('_128.npz'):
                    sample_loaded['voxel'] = np.load(v)['voxel'][None, ...]
                elif v.endswith('_spherical.npz'):
                    spherical_data = np.load(v)
                    sample_loaded['spherical_object'] = spherical_data['obj_spherical'][None, ...]
                    sample_loaded['spherical_depth'] = spherical_data['depth_spherical'][None, ...]
                elif v.endswith('.mat'):
                    # Right now .mat must be voxel_canon
                    sample_loaded['voxel_canon'] = loadmat(v)['voxel'][None, ...]
                else:
                    raise NotImplementedError(v)
            # Three identical channels for grayscale images
        if self.preproc is not None:
            sample_loaded = self.preproc(sample_loaded, mode=self.mode)
        # convert all types to float32 for better copy speed
        self.convert_to_float32(sample_loaded)
        return sample_loaded

    @staticmethod
    def convert_to_float32(sample_loaded):
        for k, v in sample_loaded.items():
            if isinstance(v, np.ndarray):
                if v.dtype != np.float32:
                    sample_loaded[k] = v.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def get_classes(self):
        return self._class_str
