cd ../..
CUDA_VISIBLE_DEVICES=0 python enhancement.py --raw_wav_path /data4/xxx/datasets/LJSpeech-1.1/wavs \
                                             --test_dir ./Datascp/LSJ/ljs_audio_text_test_filelist.txt \
                                             --enhanced_dir ./test_decode/ljs/bridgevoc \
                                             --ckpt ./ckpt/LJS/pretrained/bridgevoc_bcd_ljs_22_05k_fmax_8k_nmel80.pt \
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
                                             --sampling_rate 22050 \
                                             --n_fft 1024 \
                                             --num_mels 80 \
                                             --hop_size 256 \
                                             --win_size 1024 \
                                             --fmin 0 \
                                             --fmax 8000 \
                                             --phase_init zero \
                                             --spec_factor 0.33 \
                                             --spec_abs_exponent 0.5 \
                                             --normalize \
                                             --transform_type exponent \
                                             --beta_min 0.01 \
                                             --beta_max 20 \
                                             --bridge_type gmax \
                                             --N 4 \
                                             --sampling_type sde_first_order
                                        
                                      
                                        
                                         
                                    
                                    