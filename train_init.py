import torch
import numpy as np
from torch import nn
from val import infer, compute_miou
from MinkowskiEngine import SparseTensor
from model.minkunet import MinkUNet34C
from utils.solvers import initialize_optimizer, initialize_scheduler
import os
import glob
from scannet.scannet import get_train_loader, get_evaluate_loader
import time
import yaml
import shutil
from utils import util
from utils.util import bar, logging, Meter, savetxt, setup_seed, plot_epoch_loss, plot_epoch_miou


def random_sample(data_files, k, sample_dict=None, save_log=True):
    """
    :param data_files: list of file.
    :param k: number of points to sample.
    :param sample_dict: if provied, key is file name, value is a list of annotation points,
        this can avoid to sample thoses label points.
    """
    logging(f'random sample {k} points in each scene', save_log=save_log)
    if sample_dict is None:
        sample_dict = {}
    for i, file in enumerate(data_files):
        name = file.split('/')[-1][:12]
        pcd, _, label = torch.load(file)
        sample_list = list(range(len(pcd)))
        if name in sample_dict:
            # remove duplicate
            tmp = sample_dict[name]
            for t in tmp:
                sample_list.remove(t)
        sample_dict[name] = np.random.choice(sample_list, size=k, replace=False)
        bar(f'random sample {k} points in each scene', i + 1, len(data_files))
    print()
    return sample_dict


if __name__ == '__main__':
    CONTINUE = 1  # continue checkpoint
    EXPNAME = 'exp_self_train_active_learn'
    DEVICE = torch.device('cuda')
    CONFIGPATH = 'configs/configs_train_init.yaml'

    if CONTINUE > 0:
        # continue checkpoint
        t_stamp = str(CONTINUE)
        f = open(f'{EXPNAME}/{CONTINUE}/configs_train_init.yaml', 'r')
        cfg = yaml.load(f)
        util.LOGGING_PATH = os.path.join(EXPNAME, t_stamp, 'log.txt')
        logging('\n\n' + '+' * 50 + f'\ncontinue training checkpoint at {EXPNAME}/{t_stamp} train_init')
        setup_seed(cfg['BASE']['seed'])  # set random seed
    else:
        # create new output dir
        f = open(CONFIGPATH, 'r')
        cfg = yaml.load(f)
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg['BASE']['gpu']
        t_stamp = str(cfg['BASE']['exp_id'])
        os.makedirs(os.path.join(EXPNAME, t_stamp, 'init_pretrain'))
        shutil.copy(CONFIGPATH, os.path.join(EXPNAME, t_stamp))
        util.LOGGING_PATH = os.path.join(EXPNAME, t_stamp, 'log.txt')
        logging(f'create new result {EXPNAME}/{t_stamp} train_init')
        setup_seed(cfg['BASE']['seed'])  # set random seed
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['BASE']['gpu']

    # set save path
    path_to_save_trainlog = os.path.join(EXPNAME, t_stamp, 'init_pretrain', 'train.txt')
    path_to_save_miou = os.path.join(EXPNAME, t_stamp, 'init_pretrain', 'miou.txt')
    path_to_plot_loss_png = os.path.join(EXPNAME, t_stamp, 'init_pretrain', 'loss.png')
    path_to_plot_miou_png = os.path.join(EXPNAME, t_stamp, 'init_pretrain', 'miou.png')
    path_to_save_checkpoint = os.path.join(EXPNAME, t_stamp, 'init_pretrain', 'checkpoint')
    path_to_save_annotation_dict = os.path.join(EXPNAME, t_stamp, 'init_pretrain', 'annotation_dict')

    # todo: random sample annotation points
    data_root = cfg['DATA']['data_root']
    if CONTINUE < 0:
        k = cfg['ACTIVE']['k']
        if cfg['ACTIVE']['strategie'] == 'random':
            sample_dict = random_sample(glob.glob(f'{data_root}/*'), k)
        else:
            raise ValueError()
        torch.save(sample_dict, path_to_save_annotation_dict)
    else:
        logging(f'load {path_to_save_annotation_dict}')
        sample_dict = torch.load(path_to_save_annotation_dict)
    # todo: create dataloader
    train_loader = get_train_loader(
        cfg, data_root, 'train', sample_dict, repeat=True, augment_data=True, shuffle=True)
    val_loader = get_evaluate_loader(
        cfg, cfg['DATA']['val_data_root'], 'val', batchsize=cfg['TRAINING']['batchsize'],
        num_workers=cfg['DATA']['num_workers'], augment_data=False, shuffle=False, save_log=True)

    # todo: create model
    seg_criterion = nn.CrossEntropyLoss(ignore_index=cfg['DATA']['ignore_label'])
    model = MinkUNet34C(in_channels=3, out_channels=20).to(DEVICE)
    # cfg['TRAINING']['lr'] = 2.900965662009931e-06
    optimizer = initialize_optimizer(model.parameters(), cfg)
    scheduler = initialize_scheduler(optimizer, cfg)

    normalize_color = cfg['TRAINING']['normalize_color']
    accu_iter = cfg['TRAINING']['accu_iter']
    data_iter = train_loader.__iter__()
    is_training = True
    max_iter = cfg['TRAINING']['max_iter']
    epoch = 1
    curr_iter = 1
    best_miou = 0

    # load checkpoint
    if CONTINUE > 0:
        checkpoint = torch.load(path_to_save_checkpoint)
        epoch = checkpoint['epoch'] + 1
        curr_iter = checkpoint['curr_iter'] + 1
        best_miou = checkpoint['best_miou']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logging(f'Start from epoch {epoch} curr_iter {curr_iter} best_miou {best_miou * 100:.2f} lr {lr}', end='\n\n\n')

    # load pretrain model
    if cfg['TRAINING']['pretrain'] is not None:
        model.load_state_dict(torch.load(cfg['TRAINING']['pretrain']).state_dict())
        logging('Load pretrain model {}'.format(cfg['TRAINING']['pretrain']))
    logging(f'#classifer parameters {sum([x.nelement() for x in model.parameters()])}')

    # start training
    is_training = True
    max_iter = cfg['TRAINING']['max_iter']
    while is_training:
        model.train()
        start = time.time()
        loss_meter = Meter()
        # todo: train one epoch
        for _ in range(len(train_loader) // accu_iter):
            optimizer.zero_grad()
            model.zero_grad()
            batch_loss = 0
            # accumulate gradient (one iter)
            for sub_iter in range(accu_iter):
                coords, input, target = data_iter.next()

                # For some networks, making the network invariant to even, odd coords is important
                coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)
                # Preprocess input
                if normalize_color:
                    input[:, :3] = input[:, :3] / 255. - 0.5
                sinput = SparseTensor(input.to(DEVICE), coords.to(DEVICE))
                soutput = model(sinput)
                predictions = soutput.F  # [N, C]

                loss = seg_criterion(soutput.F, target.long().to(DEVICE))
                # Compute and accumulate gradient
                loss /= accu_iter
                batch_loss += loss.cpu().item()
                loss.backward()

            # Update number of steps
            optimizer.step()
            scheduler.step()  # keep a small lr

            torch.cuda.empty_cache()
            loss_meter.update(batch_loss)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            loginfo = f'epoch {epoch:04d} iter {curr_iter} time {(time.time() - start):.1f}s loss {loss_meter.avg:.3f} lr {lr}'
            bar(loginfo, _ + 1, len(train_loader) // accu_iter)
            curr_iter += 1
            if curr_iter >= max_iter:
                logging('stop training', end='\n\n', out_print=False)  # write down the epoch log
                is_training = False
                break
        print()

        # todo: save log
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loginfo = f'epoch {epoch:04d} iter {curr_iter} time {(time.time() - start):.1f}s loss {loss_meter.avg:.3f} lr {lr}'
        logging(loginfo, end='\n\n', out_print=False)  # write down the epoch log
        savetxt(loginfo, path_to_save_trainlog)
        plot_epoch_loss(path_to_save_trainlog, epoch_idx=1, loss_idx=7, path_to_save_png=path_to_plot_loss_png)

        # todo: evaluate
        if epoch % cfg['TRAINING']['eval_epochs'] == 0:
            miou = compute_miou(infer(model, val_loader, True, DEVICE), save_log=True)
            if best_miou < miou:
                path_to_save_bestmodel = glob.glob(
                    '{}/{}/{}/bestmodel*'.format(EXPNAME, t_stamp, 'init_pretrain'))
                if len(path_to_save_bestmodel) > 0:
                    os.remove(path_to_save_bestmodel[0])  # remove history best model
                best_miou = miou
                path_to_save_bestmodel = os.path.join(EXPNAME, t_stamp, 'init_pretrain',
                                                      f'bestmodel-{best_miou * 100:.2f}-iter_{curr_iter}.pth')
                torch.save(model.state_dict(), path_to_save_bestmodel)
            logging(f'epoch {epoch} miou {miou * 100:.2f} bestiou {best_miou * 100:.2f}', end='\n\n')
            savetxt(f'epoch {epoch} miou {miou * 100:.2f}', path_to_save_miou)
            plot_epoch_miou(path_to_save_miou, epoch_idx=1, miou_idx=3, path_to_save_png=path_to_plot_miou_png)
            print()

        # todo: save checkpoint
        torch.save({
            'epoch': epoch,
            'curr_iter': curr_iter,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_miou': best_miou
        }, path_to_save_checkpoint)

        if epoch in cfg['BASE']['save_epochs']:
            path_to_save_epoch_model = os.path.join(EXPNAME, t_stamp, 'init_pretrain', f'epoch-{epoch}.pth')
            torch.save(model.state_dict(), path_to_save_epoch_model)
        epoch += 1
