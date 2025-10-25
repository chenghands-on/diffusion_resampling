python3 generate.py --network checkpoints/pretrained_score/edm-imagenet-64x64-cond-adm.pkl \
    --outdir samples/imagenet_restart_pf_4 --sampler restart --method pf --cond 1 \
    --discriminator_ckpt checkpoints/discriminator/discriminator_imagenet.pt \
    --restart_info '18; {"0": [3, 1, 19.35, 40.79], "1": [4, 1, 1.09, 1.92], "2": [4, 4, 0.59, 1.09], "3": [4, 1, 0.30, 0.59], "4": [4, 4, 0.06, 0.30]}' \
    --S_churn 0.0 --S_min 0.01 --S_max 1.0 --S_noise 1.003 --num_particles 1 --seeds 0-7 \
    --dg_weight_1st_order 0.0 --steps 8 --batch 4 --resample_inds -1 --device cpu --num_particles 2 --resampling_method ot