# PRNet in the paper (x2)
python main.py --model PRNetx2 --scale 2 --save prnet_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# PRNet in the paper (x3) - from PRNet (x2)
#python main.py --model PRNetx3 --scale 3 --save prnet_x3 --n_resblocks 32 --n_feats 252 --res_scale 0.1 --reset 

# PRNet in the paper (x4) - from PRNet (x2)
#python main.py --model PRNetx4 --scale 4 --save prnet_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train ../experiment/prnet_x2/model/model_best.pt

# PRNet in the paper (x8) - from PRNet (x4)
#python main.py --model PRNetx8 --scale 8 --save prnet_x8 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train ../experiment/prnet_x4/model/model_best.pt

# Standard benchmarks (Ex. PRNet)
#python main.py --model PRNetx2 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 2  --n_feats 256 --n_resblocks 32 --res_scale 0.1  --pre_train ../experiment/prnet_x2/model/model_best.pt --test_only # --self_ensemble --save_results --save_gt

# Standard benchmarks (Ex. PRNet_x3)
#python main.py --model PRNetx3 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 3  --n_feats 252 --n_resblocks 32 --res_scale 0.1  --pre_train ../experiment/prnet_x3/model/model_best.pt --test_only # --self_ensemble --save_results --save_gt

# Standard benchmarks (Ex. PRNet_x4)
#python main.py --model PRNetx4 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 4  --n_feats 256 --n_resblocks 32 --res_scale 0.1  --pre_train ../experiment/prnet_x4/model/model_best.pt --test_only # --self_ensemble --save_results --save_gt

# Standard benchmarks (Ex. PRNet_x8)
#python main.py --model PRNetx8 --data_test Set5_x8+Set14_x8+B100_x8+Urban100_x8+Manga109_x8 --scale 8 --res_scale 0.1 --n_feats 256 --n_resblocks 32 --pre_train ../experiment/prnet_x8/model/model_latest_282.pt --test_only # --save_results --save_gt --self_ensemble



