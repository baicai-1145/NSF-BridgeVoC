import argparse
from argparse import ArgumentParser
import os
import platform
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_lightning import seed_everything

from div.backbones.shared import BackboneRegistry
from div.data_module import SpecsDataModule
from div.sdes import SDERegistry
from div.model import ScoreModelGAN, SinModel
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

try:
	import yaml
except ImportError:
	yaml = None

def _sanitize_logging_name(name: str) -> str:
	# Replace characters that are problematic on Windows filesystems.
	# Keep alphanumerics, dot, dash, and underscore; replace others with underscore.
	import re
	safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", name)
	# Avoid extremely long directory names
	return safe[:200]

def get_argparse_groups(parser):
	groups = {}
	for group in parser._action_groups:
		group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions}
		groups[group.title] = argparse.Namespace(**group_dict)
	return groups


if __name__ == '__main__':
	# throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
	base_parser = ArgumentParser(add_help=False)
	parser = ArgumentParser()
	for parser_ in (base_parser, parser):
		parser_.add_argument("--config", type=str, default=None,
							 help="Path to YAML config file; CLI args override values from config.")
		parser_.add_argument("--mode", default="bridge-only",
							  		choices=["bridge-only", "sin-bridge"],
									help="bridge-only: calls the ScoreModel class, for Schordinger-bride-based vocoder, \
            							sin-bridge: calls the SinModel class, for single-step Schordinger-bridge-based vocoder.")
		parser_.add_argument("--backbone_bridge", type=str,
                       choices=["none"] + BackboneRegistry.get_all_names(), 
                       default="bcd")
		parser_.add_argument("--pretrained_bridge_ckpt", 
                       default='xxxx.ckpt', 
                       help="checkpoint path for score model")

		parser_.add_argument("--sde", type=str,
                       choices=SDERegistry.get_all_names(),
                       default="bridgegan")
		parser_.add_argument("--fix_seed", type=bool, default=False,
							  help="Whether to use deterministic seed for reproducibility.")
		parser_.add_argument("--max_epochs", type=int, required=False, 
                       default=3100,
							  help="Max training epochs.")
		parser_.add_argument("--max_steps", type=int,
                       default=1000000,
							  help="Max training steps.")
		parser_.add_argument("--nolog", action='store_true', 
                       help="Turn off logging (for development purposes)")
		parser_.add_argument("--logstdout", action="store_true", 
                       help="Whether to print the stdout in a separate file")

	temp_args, _ = base_parser.parse_known_args()

	if temp_args.mode == "bridge-only":
		model_cls = ScoreModelGAN
	elif temp_args.mode == "sin-bridge":
		model_cls = SinModel

	# Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
	backbone_cls_score = BackboneRegistry.get_by_name(temp_args.backbone_bridge) if temp_args.backbone_bridge != "none" else None
	sde_class = SDERegistry.get_by_name(temp_args.sde)

	trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
	trainer_parser.add_argument("--accelerator", type=str, default="gpu", help="Supports passing different accelerator types.")
	trainer_parser.add_argument("--gpus", default="auto", help="How many gpus to use.")
	trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients.")
	trainer_parser.add_argument(
		"--val_check_interval",
		type=float,
		default=1.0,
		help="How often to run validation. If >=1, interpreted as number of training steps; if 0<val<1, interpreted as fraction of an epoch.",
	)

	# parser = pl.Trainer.add_argparse_args(parser)
	model_cls.add_argparse_args(
		parser.add_argument_group(model_cls.__name__, description=model_cls.__name__))
	sde_class.add_argparse_args(
		parser.add_argument_group("SDE", description=sde_class.__name__))

	if temp_args.backbone_bridge != "none":
		backbone_cls_score.add_argparse_args(
			parser.add_argument_group("BackboneScore", description=backbone_cls_score.__name__))
	else:
		parser.add_argument_group("BackboneScore", description="none")

	# Add data module args
	data_module_cls = SpecsDataModule
	data_module_cls.add_argparse_args(
		parser.add_argument_group("DataModule", description=data_module_cls.__name__))

	# Apply YAML config defaults if provided (CLI still has highest priority)
	if temp_args.config is not None:
		if yaml is None:
			raise ImportError("PyYAML is required when using --config. Please install it via `pip install pyyaml`.")
		with open(temp_args.config, "r", encoding="utf-8") as f:
			cfg = yaml.safe_load(f)
		if not isinstance(cfg, dict):
			raise ValueError("YAML config must have a mapping (dict) at the top level.")
		defaults = {}
		for key, value in cfg.items():
			if isinstance(value, dict):
				defaults.update(value)
			else:
				defaults[key] = value
		parser.set_defaults(**defaults)

	# Parse args and separate into groups
	args = parser.parse_args()
	arg_groups = get_argparse_groups(parser)

	# Initialize logger, trainer, model, datamodule
	if temp_args.mode == "bridge-only": 
		model = model_cls(
			backbone=args.backbone_bridge, sde=args.sde, data_module_cls=data_module_cls,
			**{
				**vars(arg_groups['ScoreModelGAN']),
				**vars(arg_groups['SDE']),
				**vars(arg_groups['BackboneScore']),
				**vars(arg_groups['DataModule'])
			},
			nolog=args.nolog
		)
		if temp_args.pretrained_bridge_ckpt is not None:
			try:
				model.load_score_model(torch.load(temp_args.pretrained_bridge_ckpt))
			except:
				pass
		logging_name = f"mode=bridge-only_sde={sde_class.__name__}_backbone={args.backbone_bridge}"
		if sde_class.__name__ == "BridgeGAN":
			bridge_type = vars(arg_groups['SDE'])['bridge_type']
			c, k = vars(arg_groups['SDE'])['c'], vars(arg_groups['SDE'])['k']
			beta_max = vars(arg_groups['SDE'])['beta_max']
			logging_name += f'_sde_type_{bridge_type}_c_{c}_k_{k}_beta_max_{beta_max}'
		for k in vars(arg_groups['ScoreModelGAN'])['loss_type_list']:
			logging_name += f'_{k}'
		logging_name = _sanitize_logging_name(logging_name)
	elif temp_args.mode == "sin-bridge":
		model = model_cls(
			backbone=args.backbone_bridge, sde=args.sde, data_module_cls=data_module_cls,
			**{
				**vars(arg_groups['SinModel']),
				**vars(arg_groups['SDE']),
				**vars(arg_groups['BackboneScore']),
				**vars(arg_groups['DataModule'])
			},
			nolog=args.nolog
		)
		if temp_args.pretrained_bridge_ckpt is not None:
			try:
				model.load_score_model(torch.load(temp_args.pretrained_bridge_ckpt))
			except:
				pass
		logging_name = f"mode=sin_sde={sde_class.__name__}_backbone={args.backbone_bridge}"
		logging_name += "_teacher_N_{}_".format(vars(arg_groups['SinModel'])['teacher_inference_N'])
		if vars(arg_groups['SinModel'])["use_omni_for_distill"]:
			logging_name += "wi_omni_"
		else:
			logging_name += "wo_omni_"
		for k in vars(arg_groups['SinModel'])['loss_type_list']:
			logging_name += f'{k}'
		logging_name += "_"
		for k in vars(arg_groups['SinModel'])['distill_loss_type_list']:
			logging_name += f"{k}"
		if not vars(arg_groups['SinModel'])["use_gan"]:
			logging_name += "_wo_gan"
		logging_name = _sanitize_logging_name(logging_name)
  
	dataset_name = vars(arg_groups['DataModule'])["dataset_name"]
	logger = TensorBoardLogger(save_dir=f"./ckpt/{dataset_name}/", name=logging_name, flush_secs=30) if not args.nolog else None

	# Callbacks
	callbacks = []
	callbacks.append(TQDMProgressBar(refresh_rate=10))
	if not args.nolog:
		callbacks.append(ModelCheckpoint(dirpath=os.path.join(logger.log_dir, "checkpoints"),
			save_last=True, save_top_k=4, monitor="ValidationPESQ", mode="max", filename='{epoch}-{pesq:.4f}'))
		callbacks.append(ModelCheckpoint(dirpath=os.path.join(logger.log_dir, "checkpoints"),
			save_top_k=4, monitor="ValidationPeriodicity", mode="min", filename='{epoch}-{periodicity:.4f}'))

	# Initialize the Trainer and the DataModule
	if args.fix_seed:
		seed_everything(seef=3407, workers=True)
		deterministic = True
	else:
		deterministic = False

	trainer_kwargs = vars(arg_groups['Trainer']).copy()
	gpus_val = trainer_kwargs.pop("gpus", None)

	if platform.system() == "Windows":
		# 单机 Windows 下避免使用 DDP，多进程 spawn 容易出各种序列化问题
		trainer_kwargs["accelerator"] = "gpu"
		trainer_kwargs["devices"] = 1
		strategy = "auto"
	else:
		if gpus_val is not None:
			trainer_kwargs["devices"] = gpus_val
		strategy = "ddp"

	trainer = pl.Trainer(
		**trainer_kwargs,
		strategy=strategy,
		logger=logger,
		log_every_n_steps=10,
		num_sanity_val_steps=0,
		callbacks=callbacks,
		max_steps=args.max_steps,
		max_epochs=args.max_epochs,
		deterministic=deterministic,
	)

	# Train model
	trainer.fit(model)
