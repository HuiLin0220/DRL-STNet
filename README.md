stage 1: train i2i model
   1.1 preparing data:

       python "/home/hln0895/DRL-STNet/Translation/data_preparation.py" --root "/data/hui/UDA/crossMoDA2021/crossmoda_training/" --output "/data/hui/UDA/crossMoDA2021/Translation/" --num_samples 50
   1.2 train models

	python "/home/hln0895/DRL-STNet/stage1.py" --config "/home/hln0895/DRL-STNet/Translation/configs/cofig_crossmoda2021.yaml"