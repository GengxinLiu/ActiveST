import torch
import yaml
import os
import shutil
import time
import glob
from val import infer, compute_miou
from MinkowskiEngine import SparseTensor
from model.minkunet import MinkUNet34C
from model.loss import DistillationLoss
from scannet.scannet import get_evaluate_loader, RandomDataST, get_self_training_loader
from utils import util
from utils.solvers import initialize_optimizer, initialize_scheduler
from utils.util import logging, setup_seed, Meter, bar, plot_epoch_loss, plot_epoch_miou, savetxt
import torch.nn.functional as F
import numpy as np
import argparse


def uncertainty_sample(uncertainty_dict, mask_dict, k, power=1):
    """Sample annotation points for each scene
    :param uncertainty_dict: a python dictionary, store the uncertainty of each point in each scene.
    :param mask_dict: a python dictionary, store the mask points of each scene(avoid quering mask points).
    :param k:  int, sample num of each scene.
    :param power:  float, use to sharp the uncertatiny distribution.
    """
    annotation_now_dict = {}
    start = time.time()
    for i, name in enumerate(uncertainty_dict.keys()):
        uncertainty = uncertainty_dict[name]
        mask = mask_dict[name]
        uncertainty[mask] = 0  # do not sample on label points
        p = (uncertainty ** power / sum(uncertainty ** power)).numpy()
        if sum(p) > 1:
            p = p / sum(p)
        try:
            annotation_now_dict[name] = np.random.choice(len(uncertainty), size=k, p=p)
        except:
            raise ValueError(f'Sample Error at i-{i} name-{name}')
        bar(f'time {time.time() - start:.2f}', i + 1, len(uncertainty_dict.keys()))

    return annotation_now_dict


def random_sample(data_files, k, exist_dict=None):
    """
    :param data_files: list of file.
    :param k: number of points to sample.
    :param exist_dict: if provied, key is file name, value is a list of annotation points,
        this can avoid to sample thoses label points.
    """
    query_dict = {}
    for i, file in enumerate(data_files):
        name = file.split('/')[-1][:12]
        pcd, _, label = torch.load(file)
        sample_list = list(range(len(pcd)))
        if exist_dict is not None and name in exist_dict:
            # remove duplicate
            tmp = exist_dict[name]
            for t in tmp:
                sample_list.remove(t)
        query_dict[name] = np.random.choice(sample_list, size=k, replace=False)
        bar(f'random sample {k} points in each scene', i + 1, len(data_files))
    print()
    return query_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default=5, type=str)
    parser.add_argument('--iteration', default=1, type=int)
    args = parser.parse_args()
    t_stamp = str(args.exp_id)
    it = args.iteration

    EXPNAME = 'exp_self_train_active_learn'
    DEVICE = torch.device('cuda')
    CONFIGPATH = 'configs/configs_self_training_active_learning.yaml'

    f = open(CONFIGPATH, 'r')
    cfg = yaml.load(f)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['BASE']['gpu']

    shutil.copy(CONFIGPATH, os.path.join(EXPNAME, t_stamp))
    if not os.path.exists(os.path.join(EXPNAME, t_stamp, f'iteration-{it}')):
        os.makedirs(os.path.join(EXPNAME, t_stamp, f'iteration-{it}'))

    util.LOGGING_PATH = os.path.join(EXPNAME, t_stamp, f'iteration-{it}', 'log.txt')
    logging(f'create new result {EXPNAME}/{t_stamp} train_self_train_active_learn')
    setup_seed(cfg['BASE']['seed'])  # set random seed

    # start iteration
    strategie_num_list = cfg['ACTIVE']['strategie_num_list']
    normalize_color = True
    t_confidence = cfg['PSEUDOLLABEL']['t_confidence']
    t_uncertainty = cfg['PSEUDOLLABEL']['t_uncertainty']

    logging(f'\n\n\t\tIteration {it}:\n\n')

    # todo: load all annotation dict
    logging('load init_pretrain annotation_dict')
    annotation_dict = torch.load(os.path.join(EXPNAME, t_stamp, 'init_pretrain', 'annotation_dict'))
    for j in range(1, it):
        path_to_load_ann_dict = os.path.join(EXPNAME, t_stamp, f'iteration-{j}', 'annotation_dict')
        if os.path.exists(path_to_load_ann_dict):
            logging(f'load iteration {j} annotation_dict')
            tmp_dict = torch.load(path_to_load_ann_dict)
            for key in annotation_dict.keys():
                if key not in tmp_dict:
                    raise ValueError(f'`{key}` not in {EXPNAME}/{t_stamp}/iteration-{j}/annotation_dict!!')
                annotation_dict[key] = np.concatenate([tmp_dict[key], annotation_dict[key]])
        else:
            logging(f'skip {path_to_load_ann_dict}')

    ####################################################################################
    #                            1. generate pseudo label                              #
    ####################################################################################
    # todo: generate pseudo label and compute uncertainty
    logging(f'\nGenerate pseudo label')
    path_to_save_pseudo_label = os.path.join(EXPNAME, t_stamp, f'iteration-{it}', 'pseudo_label')
    path_to_save_uncertainty_mask_confidence = os.path.join(EXPNAME, t_stamp, f'iteration-{it}',
                                                            'unce_mask_conf_dict.pth')

    if os.path.exists(path_to_save_pseudo_label):
        logging('Pseudo label `{}` has exist'.format(path_to_save_pseudo_label))
        logging('Reload uncertainty and label dict')
        uncertainty_dict, mask_dict, confidence_dict = torch.load(path_to_save_uncertainty_mask_confidence)
    else:
        os.makedirs(path_to_save_pseudo_label)
        randomer = RandomDataST(
            phase='train', save_log=True, data_root=cfg['DATA']['data_root'], annotation_dict=annotation_dict,
            propagate_label=cfg['DATA']['propagate_label'], segment_root=cfg['DATA']['segment_root'],
            voxel_size=cfg['DATA']['voxel_size'], ignore_label=cfg['DATA']['ignore_label'],
            **cfg['PSEUDOAUGMENTATION'])
        logging(f'generate pseudo label: t_uncertainty {t_uncertainty} t_confidence {t_confidence}')
        logging('use multi_view strategie to generate pseudo label and compute uncertainty')
        # todo: load best model
        if it == 1:
            model_root = os.path.join(EXPNAME, t_stamp, 'init_pretrain')
        else:
            model_root = os.path.join(EXPNAME, t_stamp, f'iteration-{it}')
        model_path = glob.glob(f'{model_root}/checkpoint')[0]
        logging(f'load teacher model {model_path}')
        model = MinkUNet34C(in_channels=3, out_channels=20).to(DEVICE)
        model.load_state_dict(torch.load(model_path)['model'])
        model.eval()

        with torch.no_grad():
            start = time.time()
            total_points, select_points = 0, 0
            uncertainty_dict, confidence_dict, mask_dict = {}, {}, {}
            for i in range(len(randomer)):
                probas_list, logits_list = [], []
                for j in range(cfg['PSEUDOLLABEL']['pass_time']):
                    coords, feats, target, remap_idx, name = randomer.__getitem__(i)
                    # forward
                    coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)
                    if normalize_color:
                        feats[:, :3] = feats[:, :3] / 255. - 0.5
                    sinput = SparseTensor(feats.to(DEVICE), coords.to(DEVICE))
                    soutput = model(sinput)
                    # remap
                    logits = soutput.F[remap_idx].cpu()

                    probas_list.append(F.softmax(logits, dim=1))
                    logits_list.append(logits)
                    if j == 0:
                        labels = target[remap_idx].cpu()  # remap

                # todo: compute uncertainty and confidence, which are used to filter pseudo labels and query
                std_all = torch.stack(probas_list).std(dim=0)  # [N, 20] uncertainty of each class
                confidence = torch.stack(probas_list).mean(dim=0)  # [N, 20] confidence
                max_value, max_idx = torch.max(confidence, dim=1)  # get max confidence, predict label
                max_std = std_all.gather(1, max_idx.view(-1, 1)).squeeze(1)  # [N] std of predict label

                # todo: filter pseudo label
                filter_mask = (max_value > t_confidence) * (max_std < t_uncertainty)
                filter_mask *= (labels == -100)  # filter lable points, only keep unlabel points
                filter_logits = torch.stack(logits_list).mean(dim=0)[filter_mask].cpu()  # [M, 20]
                filter_ids = torch.where(filter_mask == True)[0].cpu()  # [M] idx of pseudo label
                torch.save((filter_logits, filter_ids), f'{path_to_save_pseudo_label}/{name}.pth')

                # todo: save uncertainty and annotation for query
                uncertainty_dict[name] = max_std
                mask_dict[name] = (labels != -100)  # true for label points
                confidence_dict[name] = max_value

                total_points += len(labels)
                select_points += len(filter_ids)
                rate = select_points / total_points * 100
                bar(f'Iteration {it} time {time.time() - start:.2f} rate {rate:.2f}%', i + 1, len(randomer))
            logging(f'Iteration {it} time {time.time() - start:.2f} rate {rate:.2f}%\n')
            del probas_list, logits_list, randomer, model

        # mask_dict tell you which points should be avoid to query (points in the same segment should be avoided)
        torch.save((uncertainty_dict, mask_dict, confidence_dict), path_to_save_uncertainty_mask_confidence)

    ####################################################################################
    #                            2. query annotation points                            #
    ####################################################################################
    # todo: query new annotation points for this iteration, store in `annotation_now_dict`
    if it <= len(strategie_num_list):
        path_to_save_annotation_now_dict = os.path.join(EXPNAME, t_stamp, f'iteration-{it}', 'annotation_dict')
        if os.path.exists(path_to_save_annotation_now_dict):
            logging('Annotation dict `{}` has exist.'.format(path_to_save_annotation_now_dict))
            annotation_now_dict = torch.load(path_to_save_annotation_now_dict)
        else:
            k = strategie_num_list[it - 1]
            if cfg['ACTIVE']['strategie'] == 'uncertainty':
                logging(f'\nQuery {k} Annotation Points with uncertainty sampling')
                annotation_now_dict = uncertainty_sample(
                    uncertainty_dict, mask_dict, k, power=cfg['ACTIVE']['power'])
            elif cfg['ACTIVE']['strategie'] == 'random':
                logging(f'\nQuery {k} Annotation Points with random sampling')
                annotation_now_dict = random_sample(
                    glob.glob('{}/*'.format(cfg['DATA']['data_root'])), k, annotation_dict)
            else:
                raise NotImplementedError()
            torch.save(annotation_now_dict, path_to_save_annotation_now_dict)

        # todo: add new query points to history query points
        for key in annotation_dict.keys():
            if key not in annotation_now_dict:
                raise ValueError(f'`{key}` not in annotation_now_dict!!')
            annotation_dict[key] = np.concatenate([annotation_dict[key], annotation_now_dict[key]])
    else:
        logging('Do not query points')

    ####################################################################################
    #                            3. combine all labels                                 #
    ####################################################################################
    # todo: make self-training data loader, which will propagate all annotation points and combine pseudo labels
    train_loader = get_self_training_loader(
        cfg, cfg['DATA']['data_root'], path_to_save_pseudo_label, 'train', annotation_dict,
        repeat=True, augment_data=True, shuffle=True)
    val_loader = get_evaluate_loader(
        cfg, cfg['DATA']['val_data_root'], 'val', batchsize=cfg['TRAINING']['batchsize'],
        num_workers=cfg['DATA']['num_workers'], augment_data=False, shuffle=False, save_log=True)

    ####################################################################################
    #                            4. train a new model                                  #
    ####################################################################################
    # todo: train a new student model using pseudo labels and all query labels
    path_to_save_train = os.path.join(EXPNAME, t_stamp, f'iteration-{it}', 'train.txt')
    path_to_save_miou = os.path.join(EXPNAME, t_stamp, f'iteration-{it}', 'miou.txt')
    path_to_plot_loss_png = os.path.join(EXPNAME, t_stamp, f'iteration-{it}', 'loss.png')
    path_to_plot_miou_png = os.path.join(EXPNAME, t_stamp, f'iteration-{it}', 'miou.png')
    path_to_save_checkpoint = os.path.join(EXPNAME, t_stamp, f'iteration-{it}', 'checkpoint')
    student = MinkUNet34C(in_channels=3, out_channels=20).to(DEVICE)
    optimizer = initialize_optimizer(student.parameters(), cfg)
    scheduler = initialize_scheduler(optimizer, cfg)
    epoch = 1
    curr_iter = 1
    best_miou = 0
    kd_loss = DistillationLoss(cfg['TRAINING']['T'], cfg['TRAINING']['loss_kd'])
    data_iter = train_loader.__iter__()
    accu_iter = cfg['TRAINING']['accu_iter']
    w_gt = cfg['TRAINING']['w_gt'][it - 1]  # label weight
    w_kd = cfg['TRAINING']['w_kd'][it - 1]  # unlabeled weight
    logging(f'loss weight w_gt {w_gt}  w_kd {w_kd}')
    # checkpoint
    if os.path.exists(path_to_save_checkpoint):
        logging(f'load checkpoint from {path_to_save_checkpoint}')
        checkpoint = torch.load(path_to_save_checkpoint)
        student.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch'] + 1
        curr_iter = checkpoint['curr_iter']
        best_miou = checkpoint['best_miou']
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logging(f'epoch {epoch} curr_iter {curr_iter} lr {lr} best_miou {best_miou:.2f}')

    logging(f'#classifer parameters {sum([x.nelement() for x in student.parameters()])}')

    is_training = True
    max_iter = cfg['TRAINING']['max_iter']
    while is_training:
        student.train()
        start = time.time()
        loss_meter, loss_gt_meter, loss_kd_meter = Meter(), Meter(), Meter()

        # todo: train one epoch
        for _ in range(len(train_loader) // accu_iter):
            optimizer.zero_grad()
            student.zero_grad()
            batch_loss, batch_loss_gt, batch_loss_kd = 0, 0, 0
            # accumulate gradient (one iter)
            for sub_iter in range(accu_iter):
                coords, input, pseudo_gt_target, pseudo_masks, gt_mask = data_iter.next()
                # For some networks, making the network invariant to even, odd coords is important
                coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)
                # Preprocess input
                if normalize_color:
                    input[:, :3] = input[:, :3] / 255. - 0.5
                sinput = SparseTensor(input.to(DEVICE), coords.to(DEVICE))
                soutput = student(sinput)
                predictions = soutput.F  # [N, C]
                # loss ground-truth
                loss_gt = F.cross_entropy(predictions[gt_mask], pseudo_gt_target[gt_mask].argmax(1).to(DEVICE),
                                          ignore_index=cfg['DATA']['ignore_label'])
                # loss distillation
                loss_kd = kd_loss(predictions[pseudo_masks], pseudo_gt_target[pseudo_masks].to(DEVICE))

                # total loss
                loss = w_gt * loss_gt + w_kd * loss_kd

                # Compute and accumulate gradient
                loss /= accu_iter
                loss.backward()
                batch_loss += loss.cpu().item()
                batch_loss_gt += loss_gt.cpu().item()
                batch_loss_kd += loss_kd.cpu().item()

            # Update number of steps
            optimizer.step()
            scheduler.step()  # keep a small lr

            torch.cuda.empty_cache()
            loss_meter.update(batch_loss)
            loss_gt_meter.update(batch_loss_gt)
            loss_kd_meter.update(batch_loss_kd)
            loginfo = f'Iteration {it} epoch {epoch:03d} iter {curr_iter} time {(time.time() - start):.1f}s ' \
                      f'l {loss_meter.avg:.3f} l_gt {loss_gt_meter.avg:.3f} l_kd {loss_kd_meter.avg:.3f}'
            bar(loginfo, _ + 1, len(train_loader) // accu_iter)
            curr_iter += 1
            if curr_iter >= max_iter:
                logging('stop training', end='\n\n', out_print=False)  # write down the epoch log
                is_training = False
                break
        print()

        # todo: save log
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loginfo = f'Iteration {it} epoch {epoch:03d} iter {curr_iter} time {(time.time() - start):.1f}s ' \
                  f'loss {loss_meter.avg:.3f} l_gt {loss_gt_meter.avg:.3f} l_kd {loss_kd_meter.avg:.3f} lr {lr}'
        logging(loginfo, end='\n\n', out_print=False)  # write down the epoch log
        savetxt(loginfo, path_to_save_train)
        plot_epoch_loss(path_to_save_train, epoch_idx=3, loss_idx=9, path_to_save_png=path_to_plot_loss_png)

        # todo: evaluate
        if epoch % cfg['TRAINING']['eval_epochs'] == 0:
            miou = compute_miou(infer(student, val_loader, True, DEVICE), save_log=True)

            if best_miou < miou:
                path_to_save_bestmodel = glob.glob(
                    '{}/{}/{}/bestmodel*'.format(EXPNAME, t_stamp, f'iteration-{it}'))
                if len(path_to_save_bestmodel) > 0:
                    os.remove(path_to_save_bestmodel[0])  # remove history best model
                best_miou = miou
                path_to_save_bestmodel = os.path.join(EXPNAME, t_stamp, f'iteration-{it}',
                                                      f'bestmodel-{best_miou * 100:.2f}-iter_{curr_iter}.pth')
                torch.save(student.state_dict(), path_to_save_bestmodel)
            logging(f'Iteration {it} epoch {epoch} miou {miou * 100:.2f} bestiou {best_miou * 100:.2f}\n\n')
            savetxt(f'epoch {epoch} miou {miou * 100:.2f}', path_to_save_miou)
            plot_epoch_miou(path_to_save_miou, epoch_idx=1, miou_idx=3, path_to_save_png=path_to_plot_miou_png)
            print()

        # todo: save checkpoint
        torch.save({
            'epoch': epoch,
            'curr_iter': curr_iter,
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_miou': best_miou
        }, path_to_save_checkpoint)

        if epoch in cfg['BASE']['save_epochs']:
            path_to_save_epoch_model = os.path.join(EXPNAME, t_stamp, f'iteration-{it}', f'epoch-{epoch}.pth')
            torch.save(student.state_dict(), path_to_save_epoch_model)
        epoch += 1
