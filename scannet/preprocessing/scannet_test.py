import glob, plyfile, numpy as np, multiprocessing as mp, torch
import os

SPLIT_TXT_PATH = {
    'test': '/home/user/Desktop/code/scence/data/scannetv2/scannetv2_test.txt'
}
TEST_OUT_PATH = '/home/user/Desktop/code/scence/data/scannetv2/test'

if not os.path.exists(TEST_OUT_PATH):
    os.makedirs(TEST_OUT_PATH)

f = open(SPLIT_TXT_PATH['test'], 'r')
test_names = [file.strip() for file in f.readlines()]
print(f'Test num {len(test_names)}')

files = sorted(glob.glob('/home/user/Desktop/code/scence/data/scannetv2/scans_test/*/*_vh_clean_2.ply'))
print(f'len files {len(files)}')


def fun(fn):
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])  # [N, 3]
    colors = np.ascontiguousarray(v[:, 3:6])  # [N, 3]
    # coords = np.ascontiguousarray(v[:, :3] - v[:, :3].mean(0))  # [N, 3]
    # colors = np.ascontiguousarray(v[:, 3:6]) / 127.5 - 1  # [N, 3]
    name = fn.split('/')[-1][:12]
    torch.save((coords, colors), TEST_OUT_PATH + f'/{name}.pth')
    print(TEST_OUT_PATH + f'/{name}.pth')


p = mp.Pool(processes=mp.cpu_count())
p.map(fun, files)
p.close()
p.join()
