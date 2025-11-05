import os
import numpy as np
import librosa as lib
import torch
import math
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from soundfile import write
from os.path import join
from argparse import ArgumentParser
from div.backbones.shared import BackboneRegistry
from div.sdes import SDERegistry
from div.data_module import mel_spectrogram, inverse_mel

def spec_fwd(spec, transform_type, spec_factor, spec_abs_exponent):
    if transform_type == "exponent":
        if spec_abs_exponent != 1:
            e = spec_abs_exponent
            spec = spec.abs() ** e * torch.exp(1j * spec.angle())
        spec = spec * spec_factor 
    elif transform_type == "log":
        spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
        spec = spec * spec_factor
    elif transform_type == "none":
        spec = spec
    return spec

def spec_back(spec, transform_type, spec_factor, spec_abs_exponent):
    if transform_type == "exponent":
        spec = spec / spec_factor
        if spec_abs_exponent != 1:
            e = spec_abs_exponent
            spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
    elif transform_type == "log":
        spec = spec / spec_factor
        spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
    elif transform_type == "none":
        spec = spec
    return spec

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--use_mel_load", action="store_true",
                        help="Whether to load mel.npy for generation.")
    parser.add_argument("--raw_wav_path", type=str, required=True, default="",
                        help='Directory of the raw wavfile')
    parser.add_argument("--test_dir", type=str, required=True, default="", 
                        help='Directory containing the test data')
    parser.add_argument("--enhanced_dir", type=str, required=True, default="",
                        help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", type=str, required=True, default="",
                        help='Path to model checkpoint')
    parser.add_argument("--sde_name", type=str, required=True, default="bridgegan", help="The type of the diffuion")
    parser.add_argument("--backbone", type=str, required=True, default="bcd", help="The type of the network backbone")
    parser.add_argument("--device", type=str, required=True, default="cuda", help="Device to use for inference")
    # network params
    parser.add_argument("--nblocks", type=int, required=True, default=8,
                          help="The number of Conv2Former blocks, 6 for tiny, 8 for mid and 16 for large.")
    parser.add_argument("--hidden_channel", type=int, required=True, default=256,
                        help="The number of hidden channels, 32 for tiny, 256 for mid, and 384 for large.")
    parser.add_argument("--f_kernel_size", type=int, required=True, default=9,
                        help="Kernel size along the sub-band axis.")
    parser.add_argument("--t_kernel_size", type=int, required=True, default=11,
                        help="Kernel size along the frame axis.")
    parser.add_argument("--mlp_ratio", type=int, required=True, default=1,
                        help="MLP ratio for expansion.")
    parser.add_argument("--ada_rank", type=int, required=True, default=32,
                        help="Lora rank for ada-sola, 8 for tiny, 32 for mid, and 48 for large.")
    parser.add_argument("--ada_alpha", type=int, required=True, default=32,
                        help="Lora alpha for ada-sola, 8 for tiny, 32 for mid, and 48 for large.")
    parser.add_argument("--use_adanorm", action="store_true",
                        help="Whether to use AdaNorm strategy.")
    parser.add_argument("--causal", action="store_true",
                        help="Whether to use causal network setups.")
    # preprocess params
    parser.add_argument("--sampling_rate", type=int, required=True, default=24000, 
                        help="Sampling rate.")
    parser.add_argument("--n_fft", type=int, required=True, default=1024, 
                        help="Number of FFT bins.")
    parser.add_argument("--num_mels", type=int, required=True, default=100, 
                        help="Number of mels.")
    parser.add_argument("--hop_size", type=int, required=True, default=256,
                        help="Window hop length. 128 by default.")
    parser.add_argument("--win_size", type=int, required=True, default=1024, 
                        help="Window size, 1024 by default.")
    parser.add_argument("--fmin", type=int, default=0,
                        help="Minimum frequency for mel conversion.")
    parser.add_argument("--fmax", type=int, required=True, default=12000,
                        help="Maximum frequency for mel conversion.")
    parser.add_argument("--phase_init", type=str, choices=["random", "zero"], default="zero",
                            help="Phase initization method.")
    parser.add_argument("--spec_factor", type=float, required=True, default=0.33, 
                        help="Factor to multiply complex STFT coefficients by. 0.33 by default.")
    parser.add_argument("--spec_abs_exponent", type=float, required=True, default=0.5, 
                        help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
    parser.add_argument("--normalize", action="store_true", 
                        help="Whether to apoply the normalization strategy.")
    parser.add_argument("--transform_type", type=str, choices=["exponent", "log", "none"], default="exponent", 
                        help="Spectogram transformation for input representation.")
    parser.add_argument("--drop_last_freq", action="store_true",
                        help="Whether to drop the last frequency band to meet the exp(2) requirement.")
    # SDE params
    parser.add_argument("--beta_min", type=float, required=True, default=0.01,
                            help="Beta min")
    parser.add_argument("--beta_max", type=float, required=True, default=20,
                        help="Beta max")
    parser.add_argument("--c", type=float, required=False, default=0.4,
                        help="Noise scheduler parameter.")
    parser.add_argument("--k", type=float, required=False, default=2.6,
                        help="Noise scheduler parameter.")
    parser.add_argument("--bridge_type", type=str, required=True, default="gmax",
                        choices=["vp", "ve", "gmax"],
                        help="Type of bridge diffusion.")
    args = parser.parse_args()

    # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
    backbone_cls_score = BackboneRegistry.get_by_name(args.backbone) if args.backbone != "none" else None
    dnn = backbone_cls_score(**vars(args))
    sde_class = SDERegistry.get_by_name(args.sde_name)
    sde = sde_class(**vars(args))
    
    try:  # Method1: load .ckpt file
        nn_weights = OrderedDict()
        ckp = torch.load(args.ckpt, map_location="cpu")["state_dict"]
        for k, v in ckp.items():
            if k.startswith("dnn"):
                nn_weights[k[4:]] = v
        dnn.load_state_dict(nn_weights)
        torch.save({"generator": nn_weights}, "/data4/liandong/PROJECTS/BridgeVoc-open/ckpt/Libritts/pretrained/bridgevoc_bcd_single_libritts_24k_fmax12k_nmel100.pt")
    except:  # Method2: load .pt file
        model_pt = torch.load(args.ckpt, map_location="cpu")
        dnn.load_state_dict(model_pt["generator"])

    dnn.to(args.device)
    dnn.eval()
    print(f"Using single-step sampling procedure.")

    # Get list of noisy files
    post_str = os.path.splitext(args.test_dir)[-1]
    enhanced_dir = args.enhanced_dir + "_NFE1"
    if not os.path.exists(enhanced_dir):
        os.makedirs(enhanced_dir)

    if post_str in ['.txt', '.scp']:
        filelist = []
        lines = open(args.test_dir, 'r').readlines()
        for l in lines:
            cur_filename = l.strip()  # wav filename
            filelist.append(os.path.join(args.raw_wav_path, cur_filename))
    else:  # dir
        if not args.use_mel_load:  # wav files
            filelist = glob(f"{args.test_dir}/*.wav") + \
                       glob(f"{args.test_dir}/*/*.wav") + \
                       glob(f"{args.test_dir}/*/*/*.wav")
        else:
            filelist = glob(f"{args.test_dir}/*.npy") + \
                       glob(f"{args.test_dir}/*/*.npy") + \
                       glob(f"{args.test_dir}/*/*/*.npy")

    # Enhance files
    for noisy_file in tqdm(filelist):
        filename = os.path.split(noisy_file)[-1]
        if not args.use_mel_load:
            data, _ = lib.load(noisy_file, sr=args.sampling_rate, mono=True)
            data = torch.FloatTensor(data.astype('float32')).unsqueeze(0).to(args.device)  # （1, L)
            T_orig = data.shape[-1]

            # Normalize
            if args.normalize:
                norm_factor = torch.max(torch.abs(data)) + 1e-6
            else:
                norm_factor = 1.0

            data = data / norm_factor
            
            # Prepare DNN input
            Y = mel_spectrogram(data, 
                                n_fft=args.n_fft, 
                                num_mels=args.num_mels, 
                                sampling_rate=args.sampling_rate,
                                hop_size=args.hop_size,
                                win_size=args.win_size,
                                fmin=args.fmin,
                                fmax=args.fmax,
                                )
        else:
            Y = np.load(noisy_file)
            Y = torch.FloatTensor(Y.astype('float32')).unsqueeze(0).to(args.device)
            Y = torch.log(torch.clamp(torch.exp(Y), min=1e-5))
            T_orig = None
            
        Y = inverse_mel(Y, 
                        n_fft=args.n_fft, 
                        num_mels=args.num_mels, 
                        sampling_rate=args.sampling_rate, 
                        hop_size=args.hop_size,
                        win_size=args.win_size, 
                        fmin=args.fmin, 
                        fmax=args.fmax,
                        ).unsqueeze(1)

        # add phase
        if args.phase_init == "zero":
            phase_ = torch.zeros_like(Y).to(Y.device)
        elif args.phase_init == 'random':
            phase_ = 2 * math.pi * torch.rand_like(Y) - math.pi  # [-pi, pi) 
        Y = torch.complex(Y * torch.cos(phase_), Y * torch.sin(phase_))  # complex-tensor, (B, 1, F, T)
        if args.drop_last_freq:
            Y = Y[:, :, :-1].contiguous()

        # range-adjust
        Y = spec_fwd(Y, args.transform_type, args.spec_factor, args.spec_abs_exponent)
        Y = torch.cat([Y.real, Y.imag], dim=1)

        if args.device == "cpu":
            use_cpu = True
        else:
            use_cpu = False

        # Single-step sampling
        t = (torch.ones([Y.shape[0]]) * (1 - sde.offset)).to(Y.device)
        with torch.no_grad():
            sample = dnn(inpt=Y.to(args.device), cond=Y.to(args.device), time_cond=t)

        sample = torch.complex(sample[:, 0], sample[:, -1]).unsqueeze(1)  # (B,1,F-1,T)
        if args.drop_last_freq:
            sample_last = sample[:, :, -1].unsqueeze(-2).contiguous()  # (B, 1, 1, T)
            sample = torch.cat([sample, sample_last], dim=-2)  # (B, 1, F, T)

        # Backward transform in time domain
        sample = spec_back(sample, args.transform_type, args.spec_factor, args.spec_abs_exponent).squeeze(1)
        x_hat = torch.istft(sample, 
                            n_fft=args.n_fft, 
                            hop_length=args.hop_size, 
                            win_length=args.win_size, 
                            window=torch.hann_window(args.win_size).to(sample.device),
                            length=T_orig).cpu()
        # Renormalize
        if args.normalize:
            if not args.use_mel_load:
                x_hat = x_hat * norm_factor.cpu()

        # Write enhanced wav file
        write(join(enhanced_dir, filename), x_hat.squeeze().numpy(), args.sampling_rate)
