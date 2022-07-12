
#### Train model on ScanNet-v2
```shell script
# 1. train a model using random sample annotations
python train_init.py

# 2. run ActiveST framework
sh train.sh
```

#### Make submission to [ScanNet-v2 benchmark](http://kaldir.vc.in.tum.de/scannet_benchmark/data_efficient/)
```shell script
# 1. make predictions
python test.py

# 2. remap label idx
python make_submit.py
```

