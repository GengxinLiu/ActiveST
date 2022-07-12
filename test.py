import torch
import time
import numpy as np
from model.minkunet import MinkUNet34C
from MinkowskiEngine import SparseTensor
from utils.util import logging, bar
from scannet.scannet import load_segment, ScanNetTest, collate_fn_test
import glob
from torch.utils.data import DataLoader
import os
import shutil


def infer(model, dataloader, normalize_color=True, device=None):
    with torch.no_grad():
        model.eval()
        start = time.time()
        output_dict = {}  # save voting result, pcd, colors and labels
        for i, batch in enumerate(dataloader):
            coords, input, _, remap_index, file_names = batch
            # For some networks, making the network invariant to even, odd coords is important
            coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)
            # Preprocess input
            if normalize_color:
                input[:, :3] = input[:, :3] / 255. - 0.5
            sinput = SparseTensor(input.to(device), coords.to(device))
            # Feed forward
            soutput = model(sinput)
            batch_idxs = coords[:, 0].numpy()  # [N]
            for j, b in enumerate(np.unique(batch_idxs)):
                idxs = np.where(batch_idxs == b)[0]
                remap_idx = remap_index[j]
                logits = soutput.F[idxs][remap_idx]  # remap to original shape
                output_dict[file_names[j]] = logits.cpu().numpy()
            bar(f'time {time.time() - start:.1f}', i + 1, len(dataloader))
        print()
    return output_dict


def vote_by_segment_uncertainty_topk(predictions, segments, uncertainty=None, topk=-1, device=None):
    """Reduce predictions by segments and broadcast to each point in the segments.
    :param predictions: [N, C], ndarray or torch.Tensor, prediction of model (logits or softmax).
    :param segments: [N], ndarray or torch.Tensor, segment id for each point.
    :param uncertainty: [N], uncertainty of each point.
    :param topk: the topk point, when topk=-1, all the point will include.
    :param device:
    :return refine: [N, C] refine predictions.
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions)
    if isinstance(segments, np.ndarray):
        segments = torch.tensor(segments)
    if uncertainty is not None and isinstance(uncertainty, np.ndarray):
        uncertainty = torch.tensor(uncertainty)

    predictions = predictions.to(device)
    segments = segments.to(device)
    if uncertainty is not None:
        uncertainty = uncertainty.to(device)

    refine = torch.zeros(predictions.shape).to(device)
    for seg_id in torch.unique(segments):
        idx = torch.where(segments == seg_id)[0]
        if topk > 0 and uncertainty is not None:
            u = -uncertainty[idx]
            min_k = min(topk, len(idx))
            filter_idx = u.topk(min_k)[1]  # get `k` most certain point to represent the segment
            res = torch.mean(predictions[idx][filter_idx], dim=0)
        else:
            res = torch.mean(predictions[idx], dim=0)
        refine[idx] = res
    return refine


if __name__ == '__main__':
    batchsize = 2

    save_predict_path = 'test_pred'
    save_predict_refine_path = 'test_pred_vote'
    if os.path.exists(save_predict_path):
        shutil.rmtree(save_predict_path)
    if os.path.exists(save_predict_refine_path):
        shutil.rmtree(save_predict_refine_path)
    os.makedirs(save_predict_path)
    os.makedirs(save_predict_refine_path)

    phase = 'test'
    DEVICE = torch.device('cuda')
    data_root = ''
    segment_root = ''
    model_path = ''

    voxel_size = 0.02
    num_workers = 0
    normalize_color = True
    augment_data = True
    elastic_distortion = False
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
    random_scale = True
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    random_rotation = True
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
    random_flip = True
    ROTATION_AXIS = 'z'
    chromaticautocontrast = True
    chromatictranslation = True
    data_aug_color_trans_ratio = 0.1
    chromaticjitter = True
    data_aug_color_jitter_std = 0.05

    unaugment_loader = DataLoader(
        dataset=ScanNetTest(
            phase, save_log=False, data_root=data_root, voxel_size=voxel_size,
            ignore_label=-100, augment_data=False),
        batch_size=batchsize,
        num_workers=num_workers,
        collate_fn=collate_fn_test,
        shuffle=False)

    loader = DataLoader(
        dataset=ScanNetTest(
            phase, save_log=False, data_root=data_root, voxel_size=voxel_size,
            ignore_label=-100, augment_data=augment_data, elastic_distortion=elastic_distortion,
            ELASTIC_DISTORT_PARAMS=ELASTIC_DISTORT_PARAMS,
            random_scale=random_scale, SCALE_AUGMENTATION_BOUND=SCALE_AUGMENTATION_BOUND,
            random_rotation=random_rotation,
            ROTATION_AUGMENTATION_BOUND=ROTATION_AUGMENTATION_BOUND,
            random_flip=random_flip, ROTATION_AXIS=ROTATION_AXIS,
            chromaticautocontrast=chromaticautocontrast,
            chromatictranslation=chromatictranslation, data_aug_color_trans_ratio=data_aug_color_trans_ratio,
            chromaticjitter=chromaticjitter, data_aug_color_jitter_std=data_aug_color_jitter_std),
        batch_size=batchsize,
        num_workers=num_workers,
        collate_fn=collate_fn_test,
        shuffle=False)

    res = None

    model = MinkUNet34C(3, 20).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    # unaugment
    for i in range(2):
        print(f'unaugment use {model_path} reps {i}')
        out = infer(model, unaugment_loader, True, DEVICE)
        if res is None:
            res = out
        else:
            for k in out.keys():
                res[k] += out[k]
    # augment
    for i in range(5):
        print(f'augment use {model_path} reps {i}')
        out = infer(model, loader, True, DEVICE)
        if res is None:
            res = out
        else:
            for k in out.keys():
                res[k] += out[k]
    for i, name in enumerate(res.keys()):
        path = f'{save_predict_path}/{name}.txt'
        np.savetxt(path, res[name].argmax(-1), fmt='%d')
        bar('save without voting', i + 1, len(res.keys()))
    print()

    logging(f'vote in segment', save_log=False)
    refine = {}
    for i, f in enumerate(res.keys()):
        logits = res[f]
        path_to_segment = glob.glob('{}/{}*'.format(segment_root, f[:12]))
        if len(path_to_segment) > 0:
            segment = torch.tensor(load_segment(path_to_segment[0]))
            refine[f] = vote_by_segment_uncertainty_topk(logits, segment, None, topk=-1, device=DEVICE).cpu().numpy()
        else:
            raise ValueError('No file name {}/{}*'.format(segment_root, f))
        bar(f'vote segment', i + 1, len(res.keys()))
    print()

    for i, name in enumerate(refine.keys()):
        path = f'{save_predict_refine_path}/{name}.txt'
        np.savetxt(path, refine[name].argmax(-1), fmt='%d')
        bar('save with voting', i + 1, len(refine.keys()))
    print()
