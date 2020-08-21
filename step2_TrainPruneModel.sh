CUDA_VISIBLE_DEVICES=1 python3 prune_perfect.py --no-cuda --model checkpoint_sparsity_model_best.pth.tar --save pruned.pth.tar --percent 0.5
