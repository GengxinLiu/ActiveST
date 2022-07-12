import torch
import glob
import numpy as np
from utils.util import logging, bar
import MinkowskiEngine as ME
import scannet.transforms as t
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import json
import torch.nn.functional as F

SPLIT_TXT_PATH = {
    'train': 'PATH_TO_SCANNET/scannetv2_train.txt',
    'val': 'PATH_TO_SCANNET/scannetv2_val.txt'
}


def load_segment(path):
    f = open(path, 'r')
    seg = json.load(f)['segIndices']  # [N]
    return np.array(seg)


def load_data(path):
    """Load original data
    :return coords: [N, 3].
    :return feats: [N, 3], RGB colors(0~255).
    :return labels: [N], 0~19, -100 indicates invalid label.
    """
    coords, feats, labels = torch.load(path)[:3]
    return coords, feats, labels


class InfSampler(Sampler):
    """Samples elements randomly, without replacement.
      Arguments:
          data_source (Dataset): dataset to sample from
      """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)

    next = __next__  # Python 2 compatibility


class ProcessCoords(object):
    def __init__(
            self,
            voxel_size=0.02, ignore_label=-100,
            elastic_distortion=False, ELASTIC_DISTORT_PARAMS=((0.2, 0.4), (0.8, 1.6)),
            random_scale=True, SCALE_AUGMENTATION_BOUND=(0.9, 1.1),
            random_rotation=False,
            ROTATION_AUGMENTATION_BOUND=((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi)),
            random_flip=True, ROTATION_AXIS='z'
    ):
        input_transforms = []
        if elastic_distortion:
            input_transforms.append(t.ElasticDistortion(ELASTIC_DISTORT_PARAMS))
        if random_rotation:
            input_transforms.append(t.RandomRotation(ROTATION_AUGMENTATION_BOUND))
        if random_flip:
            input_transforms.append(t.RandomHorizontalFlip(ROTATION_AXIS, False))
        self.input_transforms = t.Compose(input_transforms)
        self.voxelizer = t.Voxelize(voxel_size, random_scale, SCALE_AUGMENTATION_BOUND, ignore_label=ignore_label)

    def __call__(self, coords, colors, labels):
        coords, colors, labels = self.input_transforms(coords, colors, labels)
        coords, colors, labels, remap_index = self.voxelizer(coords, colors, labels)
        return coords, colors, labels, remap_index


class ProcessColors(object):
    def __init__(
            self,
            chromaticautocontrast=True,
            chromatictranslation=True, data_aug_color_trans_ratio=0.1,
            chromaticjitter=True, data_aug_color_jitter_std=0.05
    ):
        input_transforms = []
        if chromaticautocontrast:
            input_transforms.append(t.ChromaticAutoContrast())
        if chromatictranslation:
            input_transforms.append(t.ChromaticTranslation(data_aug_color_trans_ratio))
        if chromaticjitter:
            input_transforms.append(t.ChromaticJitter(data_aug_color_jitter_std))
        self.input_transforms = t.Compose(input_transforms)

    def __call__(self, coords, colors, labels):
        coords, colors, labels = self.input_transforms(coords, colors, labels)
        return coords, colors, labels


#################
# evaluate data #
#################
class ScanNetEvaluate(Dataset):
    def __init__(
            self, phase='val', save_log=True, data_root='path2data',
            voxel_size=0.02, ignore_label=-100, augment_data=False,
            elastic_distortion=True, ELASTIC_DISTORT_PARAMS=((0.2, 0.4), (0.8, 1.6)),
            random_scale=True, SCALE_AUGMENTATION_BOUND=(0.9, 1.1),
            random_rotation=True,
            ROTATION_AUGMENTATION_BOUND=((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi)),
            random_flip=True, ROTATION_AXIS='z',
            chromaticautocontrast=True,
            chromatictranslation=True, data_aug_color_trans_ratio=0.1,
            chromaticjitter=True, data_aug_color_jitter_std=0.05,
            **kwargs
    ):
        logging(' * ' * 10 + 'Initialise ScanNetv2 Dataset' + ' * ' * 10, save_log=save_log)
        # load files path
        self.data_paths, self.file_names = [], []
        f = open(SPLIT_TXT_PATH[phase], 'r')
        for file in f.readlines():
            # file name, e.g. scene0191_00
            self.data_paths.append(glob.glob(f'{data_root}/*{file.strip()}*')[0])
            self.file_names.append(file[:12])

        if not augment_data:
            elastic_distortion = False
            random_scale = False
            random_rotation = False
            random_flip = False
            chromaticautocontrast = False
            chromatictranslation = False
            chromaticjitter = False

        log = f'phase {phase}\n' \
              f'data_root {data_root}\n' \
              f'voxel_size {voxel_size}\n' \
              f'augment_data {augment_data}\n' \
              f'elastic_distortion {elastic_distortion} ELASTIC_DISTORT_PARAMS {ELASTIC_DISTORT_PARAMS}\n' \
              f'random_scale {random_scale} SCALE_AUGMENTATION_BOUND {SCALE_AUGMENTATION_BOUND}\n' \
              f'random_rotation {random_rotation} ROTATION_AUGMENTATION_BOUND {ROTATION_AUGMENTATION_BOUND}\n' \
              f'random_flip {random_flip} ROTATION_AXIS {ROTATION_AXIS}\n' \
              f'chromaticautocontrast {chromaticautocontrast}\n' \
              f'chromatictranslation {chromatictranslation} data_aug_color_trans_ratio {data_aug_color_trans_ratio}\n' \
              f'chromaticjitter {chromaticjitter} data_aug_color_jitter_std {data_aug_color_jitter_std}\n'
        logging(log, save_log=save_log)

        # coords augmentation
        self.process_coords = ProcessCoords(
            voxel_size=voxel_size, ignore_label=ignore_label,
            elastic_distortion=elastic_distortion, ELASTIC_DISTORT_PARAMS=ELASTIC_DISTORT_PARAMS,
            random_scale=random_scale, SCALE_AUGMENTATION_BOUND=SCALE_AUGMENTATION_BOUND,
            random_rotation=random_rotation, ROTATION_AUGMENTATION_BOUND=ROTATION_AUGMENTATION_BOUND,
            random_flip=random_flip, ROTATION_AXIS=ROTATION_AXIS
        )
        # colors augmentation
        self.process_colors = ProcessColors(
            chromaticautocontrast=chromaticautocontrast,
            chromatictranslation=chromatictranslation, data_aug_color_trans_ratio=data_aug_color_trans_ratio,
            chromaticjitter=chromaticjitter, data_aug_color_jitter_std=data_aug_color_jitter_std
        )

    def __getitem__(self, index):
        coords, colors, labels = load_data(self.data_paths[index])
        # process coords
        coords, colors, labels, remap_index = self.process_coords(coords, colors, labels)
        # process colors
        coords, colors, labels = self.process_colors(coords, colors, labels)
        return tuple([coords, colors, labels, remap_index, self.file_names[index]])

    def __len__(self):
        return len(self.data_paths)


def collate_fn_evaluate(list_data):
    """Generates collate function for coords, feats, labels, remap_idx, filename.
    """
    coords, feats, labels, remap_idxs, file_names = list(zip(*list_data))
    coords_batch, feats_batch, labels_batch, remap_batch, file_batch = [], [], [], [], []
    for batch_id, _ in enumerate(coords):
        coords_batch.append(torch.from_numpy(coords[batch_id]).int())
        feats_batch.append(torch.from_numpy(feats[batch_id]))
        labels_batch.append(torch.from_numpy(labels[batch_id]).int())
        remap_batch.append(remap_idxs[batch_id])
        file_batch.append(file_names[batch_id])

    # Concatenate all lists
    coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)
    return coords_batch, feats_batch, labels_batch, remap_batch, file_batch


def get_evaluate_loader(
        cfg, data_root, phase, batchsize=1, num_workers=1, augment_data=False, shuffle=False, save_log=False
):
    dataset = ScanNetEvaluate(
        phase, save_log=save_log, data_root=data_root, voxel_size=cfg['DATA']['voxel_size'],
        ignore_label=cfg['DATA']['ignore_label'], augment_data=augment_data, **cfg['AUGMENTATION']
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        num_workers=num_workers,
        collate_fn=collate_fn_evaluate,
        shuffle=shuffle
    )
    return dataloader


#######################
# baseline train data #
#######################
class ScanNetDataEfficient(Dataset):
    def __init__(
            self, phase='train', save_log=True, data_root='path2data', annotation_dict=None,
            propagate_label=True, segment_root='path2segment',
            voxel_size=0.02, ignore_label=-100, augment_data=True,
            elastic_distortion=True, ELASTIC_DISTORT_PARAMS=((0.2, 0.4), (0.8, 1.6)),
            random_scale=True, SCALE_AUGMENTATION_BOUND=(0.9, 1.1),
            random_rotation=True,
            ROTATION_AUGMENTATION_BOUND=((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi)),
            random_flip=True, ROTATION_AXIS='z',
            random_dropout=True, dropout_ratio=0.2,
            chromaticautocontrast=True,
            chromatictranslation=True, data_aug_color_trans_ratio=0.1,
            chromaticjitter=True, data_aug_color_jitter_std=0.05,
            **kwargs
    ):
        logging('\n\n' + ' * ' * 10 + 'Initialise ScanNetv2 Dataset' + ' * ' * 10, save_log=save_log)
        # load files path
        self.data_paths, self.file_names = [], []
        f = open(SPLIT_TXT_PATH[phase], 'r')
        for file in f.readlines():
            # file name, e.g. scene0191_00
            self.data_paths.append(glob.glob(f'{data_root}/*{file.strip()}*')[0])
            self.file_names.append(file[:12])

        if not augment_data:
            elastic_distortion = False
            random_scale = False
            random_rotation = False
            random_flip = False
            random_dropout = False
            chromaticautocontrast = False
            chromatictranslation = False
            chromaticjitter = False

        log = f'phase {phase}\n' \
              f'data_root {data_root}\n' \
              f'propagate_label {propagate_label}\n' \
              f'segment_root {segment_root}\n' \
              f'voxel_size {voxel_size}\n' \
              f'augment_data {augment_data}\n' \
              f'elastic_distortion {elastic_distortion} ELASTIC_DISTORT_PARAMS {ELASTIC_DISTORT_PARAMS}\n' \
              f'random_scale {random_scale} SCALE_AUGMENTATION_BOUND {SCALE_AUGMENTATION_BOUND}\n' \
              f'random_rotation {random_rotation} ROTATION_AUGMENTATION_BOUND {ROTATION_AUGMENTATION_BOUND}\n' \
              f'random_flip {random_flip} ROTATION_AXIS {ROTATION_AXIS}\n' \
              f'random_dropout {random_dropout} dropout_ratio {dropout_ratio}\n' \
              f'chromaticautocontrast {chromaticautocontrast}\n' \
              f'chromatictranslation {chromatictranslation} data_aug_color_trans_ratio {data_aug_color_trans_ratio}\n' \
              f'chromaticjitter {chromaticjitter} data_aug_color_jitter_std {data_aug_color_jitter_std}\n'
        logging(log, save_log=save_log)

        # coords augmentation
        self.process_coords = ProcessCoords(
            voxel_size=voxel_size, ignore_label=ignore_label,
            elastic_distortion=elastic_distortion, ELASTIC_DISTORT_PARAMS=ELASTIC_DISTORT_PARAMS,
            random_scale=random_scale, SCALE_AUGMENTATION_BOUND=SCALE_AUGMENTATION_BOUND,
            random_rotation=random_rotation, ROTATION_AUGMENTATION_BOUND=ROTATION_AUGMENTATION_BOUND,
            random_flip=random_flip, ROTATION_AXIS=ROTATION_AXIS
        )
        # colors augmentation
        self.process_colors = ProcessColors(
            chromaticautocontrast=chromaticautocontrast,
            chromatictranslation=chromatictranslation, data_aug_color_trans_ratio=data_aug_color_trans_ratio,
            chromaticjitter=chromaticjitter, data_aug_color_jitter_std=data_aug_color_jitter_std
        )
        # random dropout
        self.random_dropout = random_dropout
        if self.random_dropout:
            self.drop = t.RandomDropout(dropout_ratio=dropout_ratio)

        # load data efficient dict
        self._make_data_efficient_label(annotation_dict, propagate_label, segment_root, save_log)

    def _make_data_efficient_label(self, limi_anno_dict, propagate_label, segment_root=None, save_log=False):
        self.sparse_labels = []
        total_annotations = 0
        for i, name in enumerate(self.file_names):
            limi_anno_idx = limi_anno_dict[name]
            _, _, labels = load_data(self.data_paths[i])
            sparse_label = np.ones((labels.shape[0]), dtype=int) * -100
            sparse_label[limi_anno_idx] = labels[limi_anno_idx]
            if propagate_label:
                # propagate label by segment
                segment = load_segment(glob.glob(f'{segment_root}/{name}*')[0])
                for idx in limi_anno_idx:
                    # points in the same segment have the same label
                    p_idx = np.where(segment == segment[idx])[0]
                    sparse_label[p_idx] = sparse_label[idx]  # propagate the label
            total_annotations += len(np.where(sparse_label != -100)[0])
            self.sparse_labels.append(sparse_label)
            bar(f'per_scenelabel {total_annotations / len(self.sparse_labels):.2f}', i + 1, len(self.file_names))
        print()
        logging(f'per_scenelabel {total_annotations / len(self.sparse_labels):.2f}\n\n', save_log=save_log)

    def __getitem__(self, index):
        coords, colors, _ = load_data(self.data_paths[index])
        labels = self.sparse_labels[index]
        # process coords
        coords, colors, labels, remap_index = self.process_coords(coords, colors, labels)
        # random dropout
        if self.random_dropout:
            coords, colors, labels, keep_inds = self.drop(coords, colors, labels)
        # process colors
        coords, colors, labels = self.process_colors(coords, colors, labels)

        return tuple([coords, colors, labels])

    def __len__(self):
        return len(self.data_paths)


def collate_fn(list_data):
    """Generates collate_function for baseline train, returan coords_batch, feats_batch, labels_batch.
    """
    coords, feats, labels = list(zip(*list_data))
    coords_batch, feats_batch, labels_batch = [], [], []
    for batch_id, _ in enumerate(coords):
        coords_batch.append(torch.from_numpy(coords[batch_id]).int())
        feats_batch.append(torch.from_numpy(feats[batch_id]))
        labels_batch.append(torch.from_numpy(labels[batch_id]).int())

    # Concatenate all lists
    coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)
    return coords_batch, feats_batch, labels_batch


def get_train_loader(
        cfg, data_root, phase, annotation_dict, repeat=True, augment_data=True, shuffle=True, save_log=True
):
    dataset = ScanNetDataEfficient(
        phase=phase, save_log=save_log, data_root=data_root, annotation_dict=annotation_dict,
        propagate_label=cfg['DATA']['propagate_label'], segment_root=cfg['DATA']['segment_root'],
        voxel_size=cfg['DATA']['voxel_size'], ignore_label=cfg['DATA']['ignore_label'], augment_data=augment_data,
        **cfg['AUGMENTATION']
    )
    if repeat:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg['TRAINING']['batchsize'],
            num_workers=cfg['DATA']['num_workers'],
            collate_fn=collate_fn,
            sampler=InfSampler(dataset, shuffle)
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg['TRAINING']['batchsize'],
            num_workers=cfg['DATA']['num_workers'],
            collate_fn=collate_fn,
            shuffle=shuffle
        )
    return dataloader


###################
# Submit Test Set #
###################
class ScanNetTest(Dataset):
    def __init__(
            self, phase='test', save_log=True, data_root='path2data',
            voxel_size=0.02, ignore_label=-100, augment_data=False,
            elastic_distortion=True, ELASTIC_DISTORT_PARAMS=((0.2, 0.4), (0.8, 1.6)),
            random_scale=True, SCALE_AUGMENTATION_BOUND=(0.9, 1.1),
            random_rotation=True,
            ROTATION_AUGMENTATION_BOUND=((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi)),
            random_flip=True, ROTATION_AXIS='z',
            chromaticautocontrast=True,
            chromatictranslation=True, data_aug_color_trans_ratio=0.1,
            chromaticjitter=True, data_aug_color_jitter_std=0.05,
            **kwargs
    ):
        logging(' * ' * 10 + 'Initialise ScanNetv2 Dataset' + ' * ' * 10, save_log=save_log)
        # load files path
        self.data_paths, self.file_names = [], []
        f = open(SPLIT_TXT_PATH[phase], 'r')
        for file in f.readlines():
            # file name, e.g. scene0191_00
            self.data_paths.append(glob.glob(f'{data_root}/*{file.strip()}*')[0])
            self.file_names.append(file[:12])

        if not augment_data:
            elastic_distortion = False
            random_scale = False
            random_rotation = False
            random_flip = False
            chromaticautocontrast = False
            chromatictranslation = False
            chromaticjitter = False

        log = f'phase {phase}\n' \
              f'data_root {data_root}\n' \
              f'voxel_size {voxel_size}\n' \
              f'augment_data {augment_data}\n' \
              f'elastic_distortion {elastic_distortion} ELASTIC_DISTORT_PARAMS {ELASTIC_DISTORT_PARAMS}\n' \
              f'random_scale {random_scale} SCALE_AUGMENTATION_BOUND {SCALE_AUGMENTATION_BOUND}\n' \
              f'random_rotation {random_rotation} ROTATION_AUGMENTATION_BOUND {ROTATION_AUGMENTATION_BOUND}\n' \
              f'random_flip {random_flip} ROTATION_AXIS {ROTATION_AXIS}\n' \
              f'chromaticautocontrast {chromaticautocontrast}\n' \
              f'chromatictranslation {chromatictranslation} data_aug_color_trans_ratio {data_aug_color_trans_ratio}\n' \
              f'chromaticjitter {chromaticjitter} data_aug_color_jitter_std {data_aug_color_jitter_std}\n'
        logging(log, save_log=save_log)

        # coords augmentation
        self.process_coords = ProcessCoords(
            voxel_size=voxel_size, ignore_label=ignore_label,
            elastic_distortion=elastic_distortion, ELASTIC_DISTORT_PARAMS=ELASTIC_DISTORT_PARAMS,
            random_scale=random_scale, SCALE_AUGMENTATION_BOUND=SCALE_AUGMENTATION_BOUND,
            random_rotation=random_rotation, ROTATION_AUGMENTATION_BOUND=ROTATION_AUGMENTATION_BOUND,
            random_flip=random_flip, ROTATION_AXIS=ROTATION_AXIS
        )
        # colors augmentation
        self.process_colors = ProcessColors(
            chromaticautocontrast=chromaticautocontrast,
            chromatictranslation=chromatictranslation, data_aug_color_trans_ratio=data_aug_color_trans_ratio,
            chromaticjitter=chromaticjitter, data_aug_color_jitter_std=data_aug_color_jitter_std
        )

    def __getitem__(self, index):
        coords, colors = torch.load(self.data_paths[index])
        labels = np.zeros(len(coords))  # ignore
        # process coords
        coords, colors, labels, remap_index = self.process_coords(coords, colors, labels)
        # process colors
        coords, colors, labels = self.process_colors(coords, colors, labels)
        return tuple([coords, colors, labels, remap_index, self.file_names[index]])

    def __len__(self):
        return len(self.data_paths)


def collate_fn_test(list_data):
    """Generates collate function for coords, feats, labels, remap_idx, filename.
    """
    coords, feats, labels, remap_idxs, file_names = list(zip(*list_data))
    coords_batch, feats_batch, labels_batch, remap_batch, file_batch = [], [], [], [], []
    for batch_id, _ in enumerate(coords):
        coords_batch.append(torch.from_numpy(coords[batch_id]).int())
        feats_batch.append(torch.from_numpy(feats[batch_id]))
        labels_batch.append(torch.from_numpy(labels[batch_id]).int())
        remap_batch.append(remap_idxs[batch_id])
        file_batch.append(file_names[batch_id])

    # Concatenate all lists
    coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)
    return coords_batch, feats_batch, labels_batch, remap_batch, file_batch


def get_test_loader(
        cfg, data_root, phase, batchsize=1, num_workers=1, augment_data=False, shuffle=False, save_log=False
):
    dataset = ScanNetEvaluate(
        phase, save_log=save_log, data_root=data_root, voxel_size=cfg['DATA']['voxel_size'],
        ignore_label=cfg['DATA']['ignore_label'], augment_data=augment_data, **cfg['AUGMENTATION']
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        num_workers=num_workers,
        collate_fn=collate_fn_evaluate,
        shuffle=shuffle
    )
    return dataloader


#######################
# self-training data# #
#######################
class RandomDataST:
    def __init__(
            self, phase='train', save_log=True, data_root='path2data', annotation_dict=None,
            propagate_label=True, segment_root='path2segment',
            voxel_size=0.02, ignore_label=-100, augment_data=True,
            elastic_distortion=True, ELASTIC_DISTORT_PARAMS=((0.2, 0.4), (0.8, 1.6)),
            random_scale=True, SCALE_AUGMENTATION_BOUND=(0.9, 1.1),
            random_rotation=True,
            ROTATION_AUGMENTATION_BOUND=((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi)),
            random_flip=True, ROTATION_AXIS='z',
            chromaticautocontrast=True,
            chromatictranslation=True, data_aug_color_trans_ratio=0.1,
            chromaticjitter=True, data_aug_color_jitter_std=0.05,
            **kwargs
    ):
        logging(' * ' * 10 + 'Initialise RandomData' + ' * ' * 10, save_log=save_log)
        # load files path
        self.data_paths, self.file_names = [], []
        f = open(SPLIT_TXT_PATH[phase], 'r')
        for file in f.readlines():
            # file name, e.g. scene0191_00
            self.data_paths.append(glob.glob(f'{data_root}/*{file.strip()}*')[0])
            self.file_names.append(file[:12])

        if not augment_data:
            elastic_distortion = False
            random_scale = False
            random_rotation = False
            random_flip = False
            chromaticautocontrast = False
            chromatictranslation = False
            chromaticjitter = False

        log = f'phase {phase}\n' \
              f'data_root {data_root}\n' \
              f'propagate_label {propagate_label}\n' \
              f'segment_root {segment_root}\n' \
              f'voxel_size {voxel_size}\n' \
              f'augment_data {augment_data}\n' \
              f'elastic_distortion {elastic_distortion} ELASTIC_DISTORT_PARAMS {ELASTIC_DISTORT_PARAMS}\n' \
              f'random_scale {random_scale} SCALE_AUGMENTATION_BOUND {SCALE_AUGMENTATION_BOUND}\n' \
              f'random_rotation {random_rotation} ROTATION_AUGMENTATION_BOUND {ROTATION_AUGMENTATION_BOUND}\n' \
              f'random_flip {random_flip} ROTATION_AXIS {ROTATION_AXIS}\n' \
              f'chromaticautocontrast {chromaticautocontrast}\n' \
              f'chromatictranslation {chromatictranslation} data_aug_color_trans_ratio {data_aug_color_trans_ratio}\n' \
              f'chromaticjitter {chromaticjitter} data_aug_color_jitter_std {data_aug_color_jitter_std}\n'
        logging(log, save_log=save_log)

        # coords augmentation
        self.process_coords = ProcessCoords(
            voxel_size=voxel_size, ignore_label=ignore_label,
            elastic_distortion=elastic_distortion, ELASTIC_DISTORT_PARAMS=ELASTIC_DISTORT_PARAMS,
            random_scale=random_scale, SCALE_AUGMENTATION_BOUND=SCALE_AUGMENTATION_BOUND,
            random_rotation=random_rotation, ROTATION_AUGMENTATION_BOUND=ROTATION_AUGMENTATION_BOUND,
            random_flip=random_flip, ROTATION_AXIS=ROTATION_AXIS
        )
        # colors augmentation
        self.process_colors = ProcessColors(
            chromaticautocontrast=chromaticautocontrast,
            chromatictranslation=chromatictranslation, data_aug_color_trans_ratio=data_aug_color_trans_ratio,
            chromaticjitter=chromaticjitter, data_aug_color_jitter_std=data_aug_color_jitter_std
        )

        # load data efficient dict
        self._make_data_efficient_label(annotation_dict, propagate_label, segment_root, save_log)

    def _make_data_efficient_label(self, limi_anno_dict, propagate_label, segment_root=None, save_log=False):
        self.sparse_labels = []
        total_annotations = 0
        for i, name in enumerate(self.file_names):
            limi_anno_idx = limi_anno_dict[name]
            _, _, labels = load_data(self.data_paths[i])
            sparse_label = np.ones((labels.shape[0]), dtype=int) * -100
            sparse_label[limi_anno_idx] = labels[limi_anno_idx]
            if propagate_label:
                # propagate label by segment
                segment = load_segment(glob.glob(f'{segment_root}/{name}*')[0])
                for idx in limi_anno_idx:
                    # points in the same segment have the same label
                    p_idx = np.where(segment == segment[idx])[0]
                    sparse_label[p_idx] = sparse_label[idx]  # propagate the label
            total_annotations += len(np.where(sparse_label != -100)[0])
            self.sparse_labels.append(sparse_label)
            bar(f'per_scenelabel {total_annotations / len(self.sparse_labels):.2f}', i + 1, len(self.file_names))
        print()
        logging(f'per_scenelabel {total_annotations / len(self.sparse_labels):.2f}', save_log=save_log)

    def __getitem__(self, index):
        coords, colors, _ = load_data(self.data_paths[index])
        labels = self.sparse_labels[index]
        # process coords
        coords, colors, labels, remap_index = self.process_coords(coords, colors, labels)
        # process colors
        coords, colors, labels = self.process_colors(coords, colors, labels)
        coords, colors, labels = ME.utils.sparse_collate([coords], [colors], [labels])
        return coords, colors, labels, remap_index, self.file_names[index]

    def __len__(self):
        return len(self.data_paths)


class ScanNetDataEfficientST(Dataset):
    def __init__(
            self, phase='train', save_log=True, data_root='path2data', pseudo_root='', annotation_dict=None,
            propagate_label=True, segment_root='path2segment',
            voxel_size=0.02, ignore_label=-100, augment_data=True,
            elastic_distortion=True, ELASTIC_DISTORT_PARAMS=((0.2, 0.4), (0.8, 1.6)),
            random_scale=True, SCALE_AUGMENTATION_BOUND=(0.9, 1.1),
            random_rotation=True,
            ROTATION_AUGMENTATION_BOUND=((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi)),
            random_flip=True, ROTATION_AXIS='z',
            random_dropout=True, dropout_ratio=0.2,
            chromaticautocontrast=True,
            chromatictranslation=True, data_aug_color_trans_ratio=0.1,
            chromaticjitter=True, data_aug_color_jitter_std=0.05,
            **kwargs
    ):
        logging(' * ' * 10 + 'Initialise ScanNetv2 Self Training Dataset' + ' * ' * 10, save_log=save_log)
        # load files path
        self.data_paths, self.file_names, self.pseudo_paths = [], [], []
        f = open(SPLIT_TXT_PATH[phase], 'r')
        for file in f.readlines():
            # file name, e.g. scene0191_00
            self.data_paths.append(glob.glob(f'{data_root}/*{file.strip()}*')[0])
            self.file_names.append(file[:12])
            self.pseudo_paths.append(glob.glob(f'{pseudo_root}/*{file.strip()}*')[0])

        if not augment_data:
            elastic_distortion = False
            random_scale = False
            random_rotation = False
            random_flip = False
            random_dropout = False
            chromaticautocontrast = False
            chromatictranslation = False
            chromaticjitter = False

        log = f'phase {phase}\n' \
              f'data_root {data_root}\n' \
              f'data_root {pseudo_root}\n' \
              f'propagate_label {propagate_label}\n' \
              f'segment_root {segment_root}\n' \
              f'voxel_size {voxel_size}\n' \
              f'augment_data {augment_data}\n' \
              f'elastic_distortion {elastic_distortion} ELASTIC_DISTORT_PARAMS {ELASTIC_DISTORT_PARAMS}\n' \
              f'random_scale {random_scale} SCALE_AUGMENTATION_BOUND {SCALE_AUGMENTATION_BOUND}\n' \
              f'random_rotation {random_rotation} ROTATION_AUGMENTATION_BOUND {ROTATION_AUGMENTATION_BOUND}\n' \
              f'random_flip {random_flip} ROTATION_AXIS {ROTATION_AXIS}\n' \
              f'random_dropout {random_dropout} dropout_ratio {dropout_ratio}\n' \
              f'chromaticautocontrast {chromaticautocontrast}\n' \
              f'chromatictranslation {chromatictranslation} data_aug_color_trans_ratio {data_aug_color_trans_ratio}\n' \
              f'chromaticjitter {chromaticjitter} data_aug_color_jitter_std {data_aug_color_jitter_std}\n'
        logging(log, save_log=save_log)

        # coords augmentation
        self.process_coords = ProcessCoords(
            voxel_size=voxel_size, ignore_label=ignore_label,
            elastic_distortion=elastic_distortion, ELASTIC_DISTORT_PARAMS=ELASTIC_DISTORT_PARAMS,
            random_scale=random_scale, SCALE_AUGMENTATION_BOUND=SCALE_AUGMENTATION_BOUND,
            random_rotation=random_rotation, ROTATION_AUGMENTATION_BOUND=ROTATION_AUGMENTATION_BOUND,
            random_flip=random_flip, ROTATION_AXIS=ROTATION_AXIS
        )
        # colors augmentation
        self.process_colors = ProcessColors(
            chromaticautocontrast=chromaticautocontrast,
            chromatictranslation=chromatictranslation, data_aug_color_trans_ratio=data_aug_color_trans_ratio,
            chromaticjitter=chromaticjitter, data_aug_color_jitter_std=data_aug_color_jitter_std
        )
        # random dropout
        self.random_dropout = random_dropout
        if self.random_dropout:
            self.drop = t.RandomDropout(dropout_ratio=dropout_ratio)

        self.ignore_label = ignore_label
        self._make_data_efficient_label(annotation_dict, propagate_label, segment_root, save_log)

    def _make_data_efficient_label(self, limi_anno_dict, propagate_label, segment_root=None, save_log=False):
        self.sparse_labels = []
        total_annotations = 0
        for i, name in enumerate(self.file_names):
            limi_anno_idx = limi_anno_dict[name]
            _, _, labels = load_data(self.data_paths[i])
            sparse_label = np.ones((labels.shape[0]), dtype=int) * -100
            sparse_label[limi_anno_idx] = labels[limi_anno_idx]
            if propagate_label:
                # propagate label by segment
                segment = load_segment(glob.glob(f'{segment_root}/{name}*')[0])
                for idx in limi_anno_idx:
                    # points in the same segment have the same label
                    p_idx = np.where(segment == segment[idx])[0]
                    sparse_label[p_idx] = sparse_label[idx]  # propagate the label
            total_annotations += len(np.where(sparse_label != -100)[0])
            self.sparse_labels.append(sparse_label)
            bar(f'per_scenelabel {total_annotations / len(self.sparse_labels):.2f}', i + 1, len(self.file_names))
        print()
        logging(f'per_scenelabel {total_annotations / len(self.sparse_labels):.2f}', save_log=save_log)

    def __getitem__(self, index):
        # coords, feats, labels: [N, x]
        # pseudo_labels: [N_p, 20], filter_idx: [N_p]:
        coords, colors, _ = load_data(self.data_paths[index])
        labels = self.sparse_labels[index]
        pseudo_labels, filter_idx = torch.load(self.pseudo_paths[index])  # torch.tensor

        pseudo_gt_target = torch.zeros((coords.shape[0], 20))  # [N, 20] both pseudo label and ground truth

        # scatter pseudo label to index(unlabel points)
        index = filter_idx.view(-1, 1).repeat(1, 20)  # [N_p ,20]
        pseudo_gt_target = pseudo_gt_target.scatter_add_(0, index, pseudo_labels)

        # scatter ground truth to index(label points)
        gt_idx = np.where(labels != self.ignore_label)[0]
        one_hot = F.one_hot(torch.from_numpy(labels[gt_idx]).long(), num_classes=20).float()  # [N_l, 20]
        index = torch.from_numpy(gt_idx).view(-1, 1).repeat(1, 20)  # [N_gt ,20]
        pseudo_gt_target = pseudo_gt_target.scatter_add_(0, index, one_hot)

        # make mask to compute loss
        pseudo_mask = np.zeros((coords.shape[0], 1), dtype=bool)  # [N, 1]
        pseudo_mask[filter_idx, 0] = True
        gt_mask = np.zeros((coords.shape[0], 1), dtype=bool)  # [N, 1]
        gt_mask[gt_idx, 0] = True

        tmp_labels = np.hstack([pseudo_gt_target, pseudo_mask, gt_mask])
        # process coords
        coords, colors, tmp_labels, remap_index = self.process_coords(coords, colors, tmp_labels)
        # random dropout
        if self.random_dropout:
            coords, colors, tmp_labels, keep_inds = self.drop(coords, colors, tmp_labels)
        # process colors
        coords, colors, tmp_labels = self.process_colors(coords, colors, tmp_labels)

        pseudo_gt_target, pseudo_mask, gt_mask = tmp_labels[:, :20], tmp_labels[:, -2], tmp_labels[:, -1]
        return tuple([coords, colors, pseudo_gt_target, pseudo_mask.astype(bool).reshape(-1, ),
                      gt_mask.astype(bool).reshape(-1, )])

    def __len__(self):
        return len(self.data_paths)


def collate_fn_self_training(list_data):
    coords, feats, labels, pseudo_masks, gt_masks = list(zip(*list_data))
    coords_batch, feats_batch, labels_batch, pseudo_mask_batch, gt_masks_batch = [], [], [], [], []
    for batch_id, _ in enumerate(coords):
        coords_batch.append(torch.from_numpy(coords[batch_id]).int())
        feats_batch.append(torch.from_numpy(feats[batch_id]))
        labels_batch.append(torch.from_numpy(labels[batch_id]).int())
        pseudo_mask_batch.append(torch.from_numpy(pseudo_masks[batch_id]))
        gt_masks_batch.append(torch.from_numpy(gt_masks[batch_id]))
    # Concatenate all lists
    coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)
    pseudo_mask_batch = torch.cat(pseudo_mask_batch, 0)
    gt_masks_batch = torch.cat(gt_masks_batch, 0)

    return coords_batch, feats_batch, labels_batch, pseudo_mask_batch, gt_masks_batch


def get_self_training_loader(
        cfg, data_root, pseudo_root, phase, annotation_dict, repeat=True, augment_data=True, shuffle=True, save_log=True
):
    dataset = ScanNetDataEfficientST(
        phase=phase, save_log=save_log, data_root=data_root, pseudo_root=pseudo_root, annotation_dict=annotation_dict,
        propagate_label=cfg['DATA']['propagate_label'],
        segment_root=cfg['DATA']['segment_root'], voxel_size=cfg['DATA']['voxel_size'],
        ignore_label=cfg['DATA']['ignore_label'], augment_data=augment_data,
        **cfg['AUGMENTATION']
    )
    if repeat:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg['TRAINING']['batchsize'],
            num_workers=cfg['DATA']['num_workers'],
            collate_fn=collate_fn_self_training,
            sampler=InfSampler(dataset, shuffle)
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg['TRAINING']['batchsize'],
            num_workers=cfg['DATA']['num_workers'],
            collate_fn=collate_fn_self_training,
            shuffle=shuffle
        )
    return dataloader
