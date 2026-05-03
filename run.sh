# train label completion U-Net
python train_label_complete.py --data_dir='YOUR_WHS_DIR' --n_epoch=300 --device='cuda:0' --tag='YOUR_TAG' --drop_rate=0.2 --sigma=4.0 --scaling=0.2

# perform label completion
python eval_label_complete.py --data_dir='YOUR_UKBB_DIR' --save_dir='./data/ukbb/' --n_frame=50 --device='cuda:0'

# train HeartFFDNet
python train_ffd.py --data_dir='./data/ukbb/' --n_epoch=10 --n_frame=50 --n_train=600 --n_valid=100 --drop_rate=0.0 --lr=1e-4 --device='cuda:0' --tag='YOUR_TAG' --w_edge=0.5 --w_curv=1.0

# run HeartFFDNet and estimate four-chamber volumes
python compute_volume.py --data_dir='./data/ukbb/' --n_frame=50 --device='cuda:0'

# train Cardiac Mesh Flow for uncoditional generation
python train_cmflow_uncond.py --data_dir='./data/ukbb/' --n_epoch=100 --n_frame=50 --n_train=600 --n_valid=100 --drop_rate=0.0 --lr=1e-4 --device='cuda:0' --tag='uncond' --sigma=1.0

# train Cardiac Mesh Flow for controllable conditional generation
python train_cmflow_cond.py --data_dir='./data/ukbb/' --n_epoch=100 --n_frame=50 --n_train=600 --n_valid=100 --drop_rate=0.0 --lr=1e-4 --device='cuda:0' --tag='cond' --sigma=1.0
