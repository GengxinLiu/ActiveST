import glob, plyfile, numpy as np, multiprocessing as mp, torch
import os

SPLIT_TXT_PATH = {
    'train': '/home/user/Desktop/code/scence/data/scannetv2/scannetv2_train.txt',
    'val': '/home/user/Desktop/code/scence/data/scannetv2/scannetv2_val.txt',
    'trainval': '/home/user/Desktop/code/scence/data/scannetv2/scannetv2_trainval.txt',
    'test': '/home/user/Desktop/code/scence/data/scannetv2/scannetv2_test.txt'
}
TRAIN_OUT_PATH = '/home/user/Desktop/code/scence/data/scannetv2/train_full'
VAL_OUT_PATH = '/home/user/Desktop/code/scence/data/scannetv2/val'
if not os.path.exists(TRAIN_OUT_PATH):
    os.makedirs(TRAIN_OUT_PATH)
if not os.path.exists(VAL_OUT_PATH):
    os.makedirs(VAL_OUT_PATH)
f = open(SPLIT_TXT_PATH['train'], 'r')
train_names = [file.strip() for file in f.readlines()]
f = open(SPLIT_TXT_PATH['val'], 'r')
val_names = [file.strip() for file in f.readlines()]
print(f'Train num {len(train_names)}  Val num {len(val_names)}')

files = sorted(glob.glob('/home/user/Desktop/code/scence/data/scannetv2_orig/scans/*/*_vh_clean_2.ply'))
files2 = sorted(glob.glob('/home/user/Desktop/code/scence/data/scannetv2_orig/scans/*/*_vh_clean_2.labels.ply'))
assert len(files) == len(files2)

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i


def fun(fn):
    fn2 = fn[:-3] + 'labels.ply'
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])  # [N, 3]
    colors = np.ascontiguousarray(v[:, 3:6])  # [N, 3]
    # coords = np.ascontiguousarray(v[:, :3] - v[:, :3].mean(0))  # [N, 3]
    # colors = np.ascontiguousarray(v[:, 3:6]) / 127.5 - 1  # [N, 3]
    a = plyfile.PlyData().read(fn2)
    w = remapper[np.array(a.elements[0]['label'])]  # label [N,]
    name = fn.split('/')[-1][:12]
    if name in train_names:
        torch.save((coords, colors, w), TRAIN_OUT_PATH + f'/{name}.pth')
        print(TRAIN_OUT_PATH + f'/{name}.pth')
    elif name in val_names:
        torch.save((coords, colors, w), VAL_OUT_PATH + f'/{name}.pth')
        print(VAL_OUT_PATH + f'/{name}.pth')
    else:
        print('ERROR!!!')
        f = open('error.txt', 'a+')
        f.write(f'{fn}\n')
        f.close()


p = mp.Pool(processes=mp.cpu_count())
p.map(fun, files)
p.close()
p.join()
