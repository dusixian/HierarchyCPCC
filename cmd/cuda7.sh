CUDA_VISIBLE_DEVICES=7 python main.py --timestamp MNIST_EMD_final --dataset MNIST --exp_name ERM --split split --task sub --cpcc 1 --emd 1 --num_workers 1  --seeds 3 --root /data/common/cindy2000_sh --ss_test 1 --coarse_ce 1;
CUDA_VISIBLE_DEVICES=7 python main.py --timestamp CIFAR_EMD_final --dataset CIFAR --exp_name ERM --split split --task sub --cpcc 1 --emd 1 --num_workers 1  --seeds 3 --root /data/common/cindy2000_sh --ss_test 1 --coarse_ce 1;
CUDA_VISIBLE_DEVICES=7 python main.py --timestamp CIFAR_SWD_final --dataset CIFAR --exp_name ERM --split split --task sub --cpcc 1 --emd 7 --num_workers 1  --seeds 3 --root /data/common/cindy2000_sh --ss_test 1 --coarse_ce 1