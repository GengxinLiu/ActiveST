import numpy as np
import os
import shutil
import utils

original_mapper = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
maper = np.ones(150) * -100
for i, x in enumerate(range(20)):
    maper[x] = original_mapper[i]

if os.path.exists('submit'):
    shutil.rmtree('submit')
os.makedirs('submit')
files = os.listdir('test_pred_vote')
for i, f in enumerate(files):
    pred = np.loadtxt(f'test_pred_vote/{f}').astype(int)
    pred = maper[pred].astype(int)
    np.savetxt(f'submit/{f}', pred, fmt='%d')
    utils.bar('', i + 1, len(files))
print()
