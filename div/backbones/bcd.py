import torch
import torch.nn as nn

from .shared import BackboneRegistry
from .bcd_utils.norm import *
from .bcd_utils.basic_unit import *


@BackboneRegistry.register("bcd")
class BCD(nn.Module):
   """Band-Aware Convolution Diffusion Network"""

   @staticmethod
   def add_argparse_args(parser):
      parser.add_argument("--nblocks", type=int, default=8,
                          help="The number of Conv2Former blocks, 6 for tiny, 8 for mid and 16 for large.")
      parser.add_argument("--input_channel", type=int, required=False, default=4,
                          help="The number of input channels.")
      parser.add_argument("--hidden_channel", type=int, default=256,
                          help="The number of hidden channels, 32 for tiny, 256 for mid, and 384 for large.")
      parser.add_argument("--f_kernel_size", type=int, default=9,
                          help="Kernel size along the sub-band axis.")
      parser.add_argument("--t_kernel_size", type=int, default=11,
                          help="Kernel size along the frame axis.")
      parser.add_argument("--mlp_ratio", type=int, default=1,
                          help="MLP ratio for expansion.")
      parser.add_argument("--ada_rank", type=int, default=32,
                          help="Lora rank for ada-sola, 8 for tiny, 32 for mid, and 48 for large.")
      parser.add_argument("--ada_alpha", type=int, default=32,
                          help="Lora alpha for ada-sola, 8 for tiny, 32 for mid, and 48 for large.")
      parser.add_argument("--ada_mode", type=str, default="sola",
                          help="AdaLN mode.")
      parser.add_argument("--act_type", type=str, required=False, default="gelu",
                          help="Activation type.")
      parser.add_argument("--pe_type", type=str, required=False, default="positional",
                          choices=["positional", "gaussian"])
      parser.add_argument("--scale", type=int, required=False, default=1000,
                          help="1000 when timestep is (0,1) else 1 for ddxm family.")
      parser.add_argument("--decode_type", type=str, required=False, default="ri",
                          help="Spectrum decoding strategy.")
      parser.add_argument("--use_adanorm", action="store_true",
                          help="Whether to use AdaNorm strategy.")
      parser.add_argument("--causal", action="store_true",
                          help="Whether to use causal network setups.")
      parser.add_argument("--highsr_band_mode", type=str, required=False, default="legacy",
                          choices=["legacy", "full_uniform", "ms_16_8_4"],
                          help="High-SR band split/merge mode. legacy=12/24/44 hard split; full_uniform=full-band uniform stride; ms_16_8_4=full_uniform + refine branches.")
      parser.add_argument("--highsr_split_mode", type=str, required=False, default="conv",
                          choices=["conv", "repack"],
                          help="For highsr_band_mode=full_uniform/ms_16_8_4, choose split/merge impl: conv=stride conv/convtranspose (legacy); repack=reversible pack/unpack (no frequency downsampling).")
      parser.add_argument("--highsr_freq_bins", type=int, required=False, default=1024,
                          help="Expected F bins after drop_last_freq for high SR (e.g. 1024 for n_fft=2048).")
      parser.add_argument("--highsr_coarse_stride_f", type=int, required=False, default=16,
                          help="Uniform stride_f for high-SR coarse branch (full_uniform/ms_16_8_4).")
      parser.add_argument("--highsr_refine8_start", type=int, required=False, default=256,
                          help="Refine-8 start bin (ms_16_8_4).")
      parser.add_argument("--highsr_refine4_start", type=int, required=False, default=672,
                          help="Refine-4 start bin (ms_16_8_4).")
      parser.add_argument("--highsr_refine_overlap", type=int, required=False, default=64,
                          help="Overlap width in bins for ramp fusion (ms_16_8_4).")
      parser.add_argument("--highsr_refine8_nblocks", type=int, required=False, default=4,
                          help="Refine-8 Conv2Former blocks (ms_16_8_4).")
      parser.add_argument("--highsr_refine4_nblocks", type=int, required=False, default=2,
                          help="Refine-4 Conv2Former blocks (ms_16_8_4).")
      return parser

   def __init__(self, 
                nblocks: int,
                hidden_channel: int,
                f_kernel_size: int,
                t_kernel_size: int,
                ada_rank: int = 16,
                ada_alpha: int = 16,
                ada_mode: str = "sola",
                mlp_ratio: int = 1,
                input_channel: int = 4,
                act_type: str = "gelu",
                pe_type: str = "positional",
                scale: int = 1000,
                decode_type: str = "ri",
                use_adanorm: bool = True,
                causal: bool = False,
                sampling_rate: int = 24000,
                highsr_band_mode: str = "legacy",
                highsr_split_mode: str = "conv",
                highsr_freq_bins: int = 1024,
                highsr_coarse_stride_f: int = 16,
                highsr_refine8_start: int = 256,
                highsr_refine4_start: int = 672,
                highsr_refine_overlap: int = 64,
                highsr_refine8_nblocks: int = 4,
                highsr_refine4_nblocks: int = 2,
                **unused_kwargs,
                ):
      super(BCD, self).__init__()
      self.nblocks = nblocks
      self.input_channel = input_channel
      self.hidden_channel = hidden_channel
      self.f_kernel_size = f_kernel_size
      self.t_kernel_size = t_kernel_size
      self.mlp_ratio = mlp_ratio
      self.ada_rank = ada_rank
      self.ada_alpha = ada_alpha
      self.ada_mode = ada_mode
      self.act_type = act_type 
      self.pe_type = pe_type
      self.scale = scale
      self.decode_type = decode_type
      self.use_adanorm = use_adanorm
      self.causal = causal
      self.sampling_rate = sampling_rate
      self.highsr_band_mode = str(highsr_band_mode).lower()
      self.highsr_split_mode = str(highsr_split_mode).lower()
      self.highsr_freq_bins = int(highsr_freq_bins)
      self.highsr_coarse_stride_f = int(highsr_coarse_stride_f)
      self.highsr_refine8_start = int(highsr_refine8_start)
      self.highsr_refine4_start = int(highsr_refine4_start)
      self.highsr_refine_overlap = int(highsr_refine_overlap)
      self.highsr_refine8_nblocks = int(highsr_refine8_nblocks)
      self.highsr_refine4_nblocks = int(highsr_refine4_nblocks)

      if self.sampling_rate > 24000:
         if self.highsr_band_mode == "legacy":
            self.enc = SharedBandSplit_NB48_HighSR(input_channel=self.input_channel,
                                                   feature_dim=self.hidden_channel,
                                                   use_adanorm=self.use_adanorm,
                                                   causal=self.causal,
                                                   )
            self.nband = self.enc.get_nband()
            self.dec = SharedBandMerge_NB48_HighSR(nband=self.nband,
                                                   feature_dim=self.hidden_channel,
                                                   use_adanorm=self.use_adanorm,
                                                   decode_type=self.decode_type)
            self.use_refine = False
         elif self.highsr_band_mode in ["full_uniform", "ms_16_8_4"]:
            if self.highsr_freq_bins % self.highsr_coarse_stride_f != 0:
               raise ValueError(
                  f"highsr_freq_bins must be divisible by highsr_coarse_stride_f, "
                  f"got {self.highsr_freq_bins} / {self.highsr_coarse_stride_f}"
               )

            if self.highsr_split_mode == "conv":
               enc_cls, dec_cls = SharedBandSplit_Uniform, SharedBandMerge_Uniform
            elif self.highsr_split_mode == "repack":
               enc_cls, dec_cls = SharedBandSplit_Repack, SharedBandMerge_Repack
            else:
               raise ValueError(f"Unknown highsr_split_mode: {self.highsr_split_mode}")

            self.enc = enc_cls(
               freq_bins=self.highsr_freq_bins,
               stride_f=self.highsr_coarse_stride_f,
               input_channel=self.input_channel,
               feature_dim=self.hidden_channel,
               use_adanorm=self.use_adanorm,
               causal=self.causal,
            )
            self.nband = self.enc.get_nband()
            self.dec = dec_cls(
               nband=self.nband,
               stride_f=self.highsr_coarse_stride_f,
               feature_dim=self.hidden_channel,
               use_adanorm=self.use_adanorm,
               decode_type=self.decode_type,
            )
            self.use_refine = self.highsr_band_mode == "ms_16_8_4"
         else:
            raise ValueError(f"Unknown highsr_band_mode: {self.highsr_band_mode}")
      else:
         self.enc = SharedBandSplit_NB24_24k(input_channel=self.input_channel,
                                             feature_dim=self.hidden_channel,
                                             use_adanorm=self.use_adanorm,
                                             causal=self.causal,
                                             )
         self.nband = self.enc.get_nband()
         self.dec = SharedBandMerge_NB24_24k(nband=self.nband,
                                             feature_dim=self.hidden_channel,
                                             use_adanorm=self.use_adanorm,
                                             decode_type=self.decode_type)
         self.use_refine = False

      if self.use_refine:
         if self.decode_type.lower() != "ri":
            raise NotImplementedError("ms_16_8_4 currently supports decode_type=ri only.")

         if self.highsr_refine_overlap <= 0:
            raise ValueError("highsr_refine_overlap must be > 0 for ms_16_8_4.")

         refine8_stride_f = 8
         refine4_stride_f = 4
         refine8_end = self.highsr_refine4_start + self.highsr_refine_overlap
         refine4_end = self.highsr_freq_bins
         refine8_len = refine8_end - self.highsr_refine8_start
         refine4_len = refine4_end - self.highsr_refine4_start
         overlap_len = refine8_end - self.highsr_refine4_start

         if refine8_len <= 0 or refine4_len <= 0:
            raise ValueError("Invalid refine slice bins: check refine8_start/refine4_start/freq_bins/overlap.")
         if refine8_len % refine8_stride_f != 0:
            raise ValueError(f"Refine-8 slice length must be divisible by {refine8_stride_f}, got {refine8_len}")
         if refine4_len % refine4_stride_f != 0:
            raise ValueError(f"Refine-4 slice length must be divisible by {refine4_stride_f}, got {refine4_len}")
         if overlap_len != self.highsr_refine_overlap:
            raise ValueError("Expected refine8_end == refine4_start + overlap.")
         if overlap_len > refine8_len or overlap_len > refine4_len:
            raise ValueError("Overlap must be <= each refine slice length.")

         self.refine8_start = self.highsr_refine8_start
         self.refine8_end = refine8_end
         self.refine4_start = self.highsr_refine4_start
         self.refine4_end = refine4_end

         if self.highsr_split_mode == "conv":
            refine_enc_cls, refine_dec_cls = SharedBandSplit_Uniform, SharedBandMerge_Uniform
         elif self.highsr_split_mode == "repack":
            refine_enc_cls, refine_dec_cls = SharedBandSplit_Repack, SharedBandMerge_Repack
         else:
            raise ValueError(f"Unknown highsr_split_mode: {self.highsr_split_mode}")

         self.refine8_enc = refine_enc_cls(
            freq_bins=refine8_len,
            stride_f=refine8_stride_f,
            input_channel=self.input_channel,
            feature_dim=self.hidden_channel,
            use_adanorm=self.use_adanorm,
            causal=self.causal,
         )
         self.refine8_dec = refine_dec_cls(
            nband=self.refine8_enc.get_nband(),
            stride_f=refine8_stride_f,
            feature_dim=self.hidden_channel,
            use_adanorm=self.use_adanorm,
            decode_type=self.decode_type,
         )
         self.refine8_net = Conv2FormerNet(nband=self.refine8_enc.get_nband(),
                                           nblocks=self.highsr_refine8_nblocks,
                                           input_channel=self.hidden_channel,
                                           hidden_channel=self.hidden_channel,
                                           f_kernel_size=self.f_kernel_size,
                                           t_kernel_size=self.t_kernel_size,
                                           mlp_ratio=self.mlp_ratio,
                                           ada_rank=self.ada_rank,
                                           ada_alpha=self.ada_alpha,
                                           ada_mode=self.ada_mode,
                                           act_type=self.act_type,
                                           causal=self.causal,
                                           use_adanorm=self.use_adanorm)

         self.refine4_enc = refine_enc_cls(
            freq_bins=refine4_len,
            stride_f=refine4_stride_f,
            input_channel=self.input_channel,
            feature_dim=self.hidden_channel,
            use_adanorm=self.use_adanorm,
            causal=self.causal,
         )
         self.refine4_dec = refine_dec_cls(
            nband=self.refine4_enc.get_nband(),
            stride_f=refine4_stride_f,
            feature_dim=self.hidden_channel,
            use_adanorm=self.use_adanorm,
            decode_type=self.decode_type,
         )
         self.refine4_net = Conv2FormerNet(nband=self.refine4_enc.get_nband(),
                                           nblocks=self.highsr_refine4_nblocks,
                                           input_channel=self.hidden_channel,
                                           hidden_channel=self.hidden_channel,
                                           f_kernel_size=self.f_kernel_size,
                                           t_kernel_size=self.t_kernel_size,
                                           mlp_ratio=self.mlp_ratio,
                                           ada_rank=self.ada_rank,
                                           ada_alpha=self.ada_alpha,
                                           ada_mode=self.ada_mode,
                                           act_type=self.act_type,
                                           causal=self.causal,
                                           use_adanorm=self.use_adanorm)

         self.alpha8 = nn.Parameter(torch.ones([1, self.hidden_channel, self.refine8_enc.get_nband(), 1]))
         self.alpha4 = nn.Parameter(torch.ones([1, self.hidden_channel, self.refine4_enc.get_nband(), 1]))
         with torch.no_grad():
            self.alpha8.mul_(1e-4)
            self.alpha4.mul_(1e-4)

         ramp = torch.linspace(0.0, 1.0, overlap_len, dtype=torch.float32)
         w8 = torch.ones([refine8_len], dtype=torch.float32)
         w8[:overlap_len] = ramp
         w8[-overlap_len:] = 1.0 - ramp
         w4 = torch.ones([refine4_len], dtype=torch.float32)
         w4[:overlap_len] = ramp
         self.register_buffer("refine8_weight", w8.reshape(1, 1, -1, 1), persistent=False)
         self.register_buffer("refine4_weight", w4.reshape(1, 1, -1, 1), persistent=False)

      self.main_net = Conv2FormerNet(nband=self.nband,
                                    nblocks=self.nblocks,
                                    input_channel=self.hidden_channel,
                                    hidden_channel=self.hidden_channel,
                                    f_kernel_size=self.f_kernel_size,
                                    t_kernel_size=self.t_kernel_size,
                                    mlp_ratio=self.mlp_ratio,
                                    ada_rank=self.ada_rank,
                                    ada_alpha=self.ada_alpha,
                                    ada_mode=self.ada_mode,
                                    act_type=self.act_type,
                                    causal=self.causal,
                                    use_adanorm=self.use_adanorm,
                                    )
      
      if self.pe_type == "positional":
         self.time_embed = PositionalTimestepEmbedder(hidden_size=self.hidden_channel, scale=self.scale)
      elif self.pe_type == "gaussian":
         self.time_embed = GaussianFourierProjection(embedding_size=self.hidden_channel)
         
      if self.ada_mode.lower() in ["vanilla", "single", "sola"] and self.use_adanorm:
         self.time_act = nn.SiLU()
         self.time_ada_final_nn1 = nn.Linear(self.hidden_channel, 2 * self.hidden_channel, bias=True)
         self.time_ada_final_nn2 = nn.Linear(self.hidden_channel, 2 * self.hidden_channel, bias=True)
         self.time_ada_begin_nn = nn.Linear(self.hidden_channel, 2 * self.hidden_channel, bias=True)
         if self.ada_mode.lower() in ["single", "sola"]:
            self.time_ada_nn = nn.Linear(self.hidden_channel, 6 * self.hidden_channel, bias=True)
         else:
            self.time_ada_nn = None

      self.alpha = nn.Parameter(torch.ones([1, self.hidden_channel, self.nband, 1]))
      with torch.no_grad():
         self.alpha.mul_(1e-4)

      self.initialize_weights()

   def initialize_weights(self):
      def _basic_init(module):
         if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
               nn.init.constant_(module.bias, 0)
      self.apply(_basic_init)

      # Zero-out AdaLN
      if self.use_adanorm:
         self._init_ada()

   def _init_ada(self):
      if self.ada_mode.lower() in ["single", "sola"]:
         nn.init.constant_(self.time_ada_nn.weight, 0)
         nn.init.constant_(self.time_ada_nn.bias, 0)
      nn.init.constant_(self.time_ada_final_nn1.weight, 0)
      nn.init.constant_(self.time_ada_final_nn1.bias, 0)
      nn.init.constant_(self.time_ada_final_nn2.weight, 0)
      nn.init.constant_(self.time_ada_final_nn2.bias, 0)
      nn.init.constant_(self.time_ada_begin_nn.weight, 0)
      nn.init.constant_(self.time_ada_begin_nn.bias, 0)
      if self.ada_mode.lower() == "vanilla":
         for block in self.main_net.net:
            nn.init.constant_(block.ada.time_nn.weight, 0)
            nn.init.constant_(block.ada.time_nn.bias, 0)
      elif self.ada_mode.lower() == "sola":
         for block in self.main_net.net:
            nn.init.kaiming_uniform_(block.ada.lora_a.weight, a=math.sqrt(5))
            nn.init.constant_(block.ada.lora_b.weight, 0) 
         if getattr(self, "use_refine", False):
            for block in self.refine8_net.net:
               nn.init.kaiming_uniform_(block.ada.lora_a.weight, a=math.sqrt(5))
               nn.init.constant_(block.ada.lora_b.weight, 0)
            for block in self.refine4_net.net:
               nn.init.kaiming_uniform_(block.ada.lora_a.weight, a=math.sqrt(5))
               nn.init.constant_(block.ada.lora_b.weight, 0)

   def forward(self, inpt, cond=None, time_cond=None):
      """
      inpt: (B, 2, F, T)
      cond: (B, 2, F, T)
      time_cond: (B,)
      return: (B, 2, F, T)
      """
      if time_cond.ndim < 1:
         time_cond = time_cond.unsqueeze(0)
      time_token = self.time_embed(time_cond)
      time_ada, time_ada_begin, time_ada_final1, time_ada_final2 = None, None, None, None
      if self.use_adanorm:
         time_token = self.time_act(time_token)
         if self.time_ada_nn is not None:
            time_ada = self.time_ada_nn(time_token)
         time_ada_final1 = self.time_ada_final_nn1(time_token)
         time_ada_final2 = self.time_ada_final_nn2(time_token)
         time_ada_begin = self.time_ada_begin_nn(time_token)

      inpt_spec = torch.cat([inpt, cond], dim=1)
      # band split
      enc_x = self.enc(inpt_spec, time_ada_begin=time_ada_begin)
      x = enc_x
      # sub-band modeling
      x = self.main_net(enc_x, time_token=time_token, time_ada=time_ada)

      x = x + self.alpha * enc_x

      # band merge, different reconstrcution strategies
      if self.decode_type.lower() == "mag+phase":
         cur_mag, cur_pha = self.dec(x, time_ada_final1=time_ada_final1, time_ada_final2=time_ada_final2)
         out_real, out_imag = cur_mag * torch.cos(cur_pha), cur_mag * torch.sin(cur_pha)
         out = torch.stack([out_real, out_imag], dim=1)
      elif self.decode_type.lower() == "ri":
         out_real, out_imag = self.dec(x, time_ada_final1=time_ada_final1, time_ada_final2=time_ada_final2)
         out = torch.cat([out_real, out_imag], dim=1)
      else:
         raise NotImplementedError("Only mag+phase and ri are supported, please check it carefully!")

      if getattr(self, "use_refine", False):
         if inpt_spec.shape[-2] != self.highsr_freq_bins:
            raise ValueError(f"Expected F={self.highsr_freq_bins} for ms_16_8_4, got F={inpt_spec.shape[-2]}")

         x8_in = inpt_spec[..., self.refine8_start:self.refine8_end, :]
         enc8 = self.refine8_enc(x8_in, time_ada_begin=time_ada_begin)
         x8 = self.refine8_net(enc8, time_token=time_token, time_ada=time_ada)
         x8 = x8 + self.alpha8 * enc8
         d8_r, d8_i = self.refine8_dec(x8, time_ada_final1=time_ada_final1, time_ada_final2=time_ada_final2)

         out[:, 0:1, self.refine8_start:self.refine8_end, :].add_(self.refine8_weight * d8_r)
         out[:, 1:2, self.refine8_start:self.refine8_end, :].add_(self.refine8_weight * d8_i)

         x4_in = inpt_spec[..., self.refine4_start:self.refine4_end, :]
         enc4 = self.refine4_enc(x4_in, time_ada_begin=time_ada_begin)
         x4 = self.refine4_net(enc4, time_token=time_token, time_ada=time_ada)
         x4 = x4 + self.alpha4 * enc4
         d4_r, d4_i = self.refine4_dec(x4, time_ada_final1=time_ada_final1, time_ada_final2=time_ada_final2)

         out[:, 0:1, self.refine4_start:self.refine4_end, :].add_(self.refine4_weight * d4_r)
         out[:, 1:2, self.refine4_start:self.refine4_end, :].add_(self.refine4_weight * d4_i)

      return out
