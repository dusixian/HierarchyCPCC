CUDA_VISIBLE_DEVICES=2 python main.py --timestamp BREEDS_EMD_final --dataset BREEDS --exp_name ERM --split split --task sub --cpcc 1 --emd 1 --num_workers 12  --seeds 3 --root /data/common/cindy2000_sh --ss_test 1 --coarse_ce 1;
CUDA_VISIBLE_DEVICES=2 python main.py --timestamp BREEDS_Sinkhorn_final --dataset BREEDS --exp_name ERM --split split --task sub --cpcc 1 --emd 2 --num_workers 12  --seeds 3 --root /data/common/cindy2000_sh --ss_test 1 --coarse_ce 1

