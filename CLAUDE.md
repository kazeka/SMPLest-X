# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

SMPLest-X is a PyTorch framework for expressive human pose and shape estimation (EHPS) from monocular images/videos. It predicts SMPL-X body model parameters (body pose, hand pose, face expression, shape) from cropped person images. This is a research codebase, not a library — it is used for training, testing, and inference.

## Environment Setup

```bash
bash scripts/install.sh
# Conda env name: smplestx, Python 3.8, PyTorch 1.12, CUDA 11.3
```

The environment requires OSMesa for headless rendering (`libosmesa6`). On non-Debian systems you'll need to adapt the `apt-get` lines in `install.sh`.

## Commands

All scripts set `PYTHONPATH=../:$PYTHONPATH`. When invoking Python directly from the project root, use `PYTHONPATH=.` instead.

**Inference** (video or image → rendered SMPL-X overlay):
```bash
sh scripts/inference.sh {MODEL_DIR} {FILE_NAME} {FPS}
# e.g.: sh scripts/inference.sh smplest_x_h test_video.mp4 30
# Input: ./demo/{FILE_NAME}  |  Output: ./demo/result_{FILE_NAME}
# MODEL_DIR resolves to ./pretrained_models/{MODEL_DIR}/
# Add --multi_person flag in main/inference.py args for multi-person inference
```

The inference script extracts frames via `ffmpeg`, runs frame-by-frame SMPL-X estimation, then re-encodes to video. Pretrained YOLO (`yolov8x.pt`) is downloaded automatically on first use.

**Training** (multi-GPU DDP via `torch.distributed.launch`):
```bash
bash scripts/train.sh {JOB_NAME} {NUM_GPUS} {CONFIG_FILE}
# e.g.: bash scripts/train.sh smplest_x_h 16 config_smplest_x_h.py
# CONFIG_FILE is a filename under ./configs/ (not a full path)
# Outputs saved to ./outputs/train_{JOB_NAME}_{DATETIME}/
```

**Testing** (single GPU, evaluates a saved checkpoint):
```bash
sh scripts/test.sh {TEST_DATASET} {MODEL_DIR} {CKPT_ID}
# e.g.: sh scripts/test.sh SynHand smplest_x_h 5
# MODEL_DIR resolves to ./outputs/{MODEL_DIR}/
# Config is read from ./outputs/{MODEL_DIR}/code/config_base.py (not ./configs/)
# CKPT_ID is the epoch number of the snapshot file
```

**Direct Python invocation:**
```bash
PYTHONPATH=. python main/train.py --num_gpus 1 --exp_name my_run --config config_smplest_x_h.py
PYTHONPATH=. python main/test.py --num_gpus 1 --result_path smplest_x_h --ckpt_idx 5 --testset SynHand
PYTHONPATH=. python main/inference.py --num_gpus 1 --file_name my_video --ckpt_name smplest_x_h --end 300
```

## Architecture

### Data flow

```
Input image → YOLO detection → crop per person → ViT encoder → TransformerDecoderHead → SMPL-X layer → 3D mesh + joints
```

### Key modules

**`models/SMPLest_X.py` — `Model`**
The top-level `nn.Module`. `forward(inputs, targets, meta_info, mode)` returns a loss dict in `'train'` mode or an output dict in `'test'`/`'inference'` mode. It owns the SMPL-X layer and computes all losses. `get_model(cfg, mode)` is the entry point — it builds `ViT` + `TransformerDecoderHead`, loading ViTPose weights for training only.

**`models/module.py` — `ViT` + `TransformerDecoderHead`**
- `ViT`: ViTPose-style Vision Transformer encoder. Prepends `task_tokens_num=80` learnable task tokens to image patches. Returns `(spatial_features [B,C,H,W], task_tokens [B,80,C])`.
- `TransformerDecoderHead`: Cross-attention transformer decoder. Takes the 80 task tokens and predicts SMPL-X parameters via per-part linear heads (body, left/right hand, face).

**`main/config.py` — `Config`**
A `dict` subclass with dot-notation access. Configs are Python files that define a `config = {...}` dict. Loaded via `Config.load_config(path)`, updated with `cfg.update_config(new_dict)`, dumped with `cfg.dump_config()`.

**`main/base.py` — `Trainer` / `Tester`**
Wrappers around dataset loading and model init. `Trainer` handles DDP setup, optimizer (Adam + CosineAnnealingLR), and checkpoint saving. `Tester` handles single-GPU evaluation. Dataset classes are dynamically imported via `importlib` as `datasets.{name}.{name}`.

**`datasets/humandata.py` — `HumanDataset`**
Base dataset class for the HumanData format (`.npz` annotation files). Subclassed per dataset (e.g., `datasets/SynHand.py`). Supports numpy-based caching (`data/cache/`) to speed up repeated loading. The `Cache` class lazily loads `.npz` files on first access.

**`human_models/human_models.py` — `SMPL` / `SMPLX`**
Singleton wrappers (via `get_instance()`) around the `smplx` library. Store joint index mappings, body part groupings (`joint_part`), and the neutral/male/female `smplx` layers. Must be initialized once with the path to downloaded model files (`./human_models/human_model_files/`).

### Rotation representation
All predicted poses use **rot6d** (6D rotation) internally. Converted to axis-angle for the SMPL-X forward pass via `rot6d_to_axis_angle` in `utils/transforms.py`, and to rotation matrices via `batch_rodrigues` for loss computation.

### Config structure
The config dict has six top-level keys: `data`, `train`, `test`, `inference`, `model`, and `log`. The `model.encoder_config` and `model.decoder_config` dicts are passed directly as kwargs to `ViT` and `TransformerDecoderHead`. Loss weights live under the `train` key (e.g., `smplx_kps_3d_weight`).

Key config options to be aware of:
- `data.data_strategy`: `"balance"` (default, resamples all datasets to equal length) or `"concat"` (concatenates as-is)
- `data.total_data_len`: total samples per epoch when using `"balance"` strategy (`"auto"` or integer)
- `train.continue_train` + `train.start_over`: resume from checkpoint (`continue_train=True, start_over=False` continues optimizer state and epoch; `start_over=True` resets epoch to 0)
- `train.hand_loss`: enables additional 3D hand keypoint alignment losses

### Required external assets (not in repo)
- `./human_models/human_model_files/` — SMPL-X and SMPL model files
- `./pretrained_models/{name}/{name}.pth.tar` + `config_base.py` — pretrained checkpoint + its config (used by inference)
- `./pretrained_models/vitpose_huge.pth` — ViTPose encoder weights (training only)
- `./data/annot/`, `./data/img/` — HumanData annotation files and images
- `./data/cache/` — auto-generated cache files (set `use_cache: True` in config)

### Adding a new dataset
1. Create `datasets/MyDataset.py` subclassing `HumanDataset`
2. Set `self.annot_path`, `self.img_dir`, `self.annot_path_cache` in `__init__`; call `self.load_data()` or `self.load_cache()` depending on `self.use_cache`
3. Add dataset name to `trainset_humandata` list in the config
4. The `base.py` `Trainer._make_batch_generator()` dynamically imports `datasets.{name}.{name}`

Data preparation tools are in `humandata_prep/`.
