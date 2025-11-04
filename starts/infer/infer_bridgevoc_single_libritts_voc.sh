cd ../..
CUDA_VISIBLE_DEVICES=0 python enhancement_single.py --raw_wav_path /data4/xxx/datasets/LibriTTS/LibriTTS \
                                                    --test_dir /data4/xxx/datasets/LibriTTS/LibriTTS/dev-clean-other-208-tonglei \
                                                    --enhanced_dir ./test_decode/libritts/bridgevoc\
                                                    --ckpt ./ckpt/Libritts/pretrained/bridgevoc_bcd_single_libritts_24k_fmax12k_nmel100.pt \
                                                    --sde_name bridgegan \
                                                    --backbone bcd \
                                                    --device cuda \
                                                    --nblocks 8 \
                                                    --hidden_channel 256 \
                                                    --f_kernel_size 9 \
                                                    --t_kernel_size 11 \
                                                    --mlp_ratio 1 \
                                                    --ada_rank 32 \
                                                    --ada_alpha 32 \
                                                    --use_adanorm \
                                                    --sampling_rate 24000 \
                                                    --n_fft 1024 \
                                                    --num_mels 100 \
                                                    --hop_size 256 \
                                                    --win_size 1024 \
                                                    --fmin 0 \
                                                    --fmax 12000 \
                                                    --phase_init zero \
                                                    --spec_factor 0.33 \
                                                    --spec_abs_exponent 0.5 \
                                                    --normalize \
                                                    --transform_type exponent \
                                                    --beta_min 0.01 \
                                                    --beta_max 20 \
                                                    --bridge_type gmax
                                        
                                      
                                        
                                         
                                    
                                    