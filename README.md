
stage 0 preparing data for training i2i model:
       
       python "/home/hln0895/DRL-STNet/stage0.py" --config "/home/hln0895/DRL-STNet/Translation/cofig_crossmoda2021.yaml"

stage 1: train i2i model
	
 python "/home/hln0895/DRL-STNet/stage1.py" --config "/home/hln0895/DRL-STNet/Translation/cofig_crossmoda2021.yaml"

stage 2: generate fake target scans (MRI)
