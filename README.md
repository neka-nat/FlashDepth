# FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution

[**Paper**](https://arxiv.org/abs/2504.07093) | [**Project Page**](https://eyeline-labs.github.io/FlashDepth/) <br>


This repository contains the official implementation of <br>
**FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution (ICCV 2025 Highlight)** 

## WebCam Demo

```bash
uv sync
uv run python webcam_inference.py --config-path configs/flashdepth --checkpoint configs/flashdepth/iter_43002.pth --device cuda
```

## Installation
We recommend creating a [conda](https://www.anaconda.com/) environment then installing the required packages using our `setup_env.sh` script. Note that the mamba package should be installed from our local folder and the torch version should be 2.4 (as of early May 2025, Mamba2 does not work if compiling torch 2.5 and above).

```
conda create -n flashdepth python=3.11 --yes
conda activate flashdepth
bash setup_env.sh
```

## Downloading Pretrained Models 
We provide three checkpoints on huggingface. They correspond to [FlashDepth (Full)](https://huggingface.co/Eyeline-Labs/FlashDepth/tree/main/flashdepth), [FlashDepth-L](https://huggingface.co/Eyeline-Labs/FlashDepth/tree/main/flashdepth-l), and [FlashDepth-S](https://huggingface.co/Eyeline-Labs/FlashDepth/tree/main/flashdepth-s), respectively, as referenced in the paper. Generally, FlashDepth-L is most accurate and FlashDepth (Full) is fastest, but we recommend using FlashDepth-L when the input resolution is low (e.g. short side less than 518).

Save the checkpoints to `configs/flashdepth/iter_43002.pth`, `configs/flashdepth-l/iter_10001.pth`, and `configs/flashdepth-s/iter_14001.pth`, respectively. 

## Inference 
To run inference on a video:
```
torchrun train.py --config-path configs/flashdepth inference=true eval.random_input=<path to video> eval.outfolder=output
```
The output depth maps (as npy files) and mp4s will be saved to `configs/flashdepth/output/`. Change the configs path to use another model. We provide some examples: 
```
torchrun train.py --config-path configs/flashdepth inference=true eval.random_input=examples/video1.mp4 eval.outfolder=output
torchrun train.py --config-path configs/flashdepth inference=true eval.random_input=examples/video2.mp4 eval.outfolder=output
```

**If you run into `TypeError: Invalid NaN comparison` errors, add `eval.compile=false` to the command.


## Training
As reported in the paper, training is split into two stages. We first train FlashDepth-L and FlashDepth-S at resolution 518x518. Then, we train FlashDepth (Full) at higher resolution. 
To train the first stage, download the [Depth Anything V2](https://depth-anything-v2.github.io/) checkpoints and save them to `checkpoints`.

For data, see `dataloaders/README.md` to verify the data format. We generally used the default format downloaded from the official websites. Also check the dataloader python files to verify how the data is loaded, if needed.
```
# first stage 
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth-l/ load=checkpoints/depth_anything_v2_vitl.pth dataset.data_root=<path to data>
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth-s/ load=checkpoints/depth_anything_v2_vits.pth dataset.data_root=<path to data>

# second stage 
torchrun --nproc_per_node=8 train.py --config-path configs/flashdepth load=configs/flashdepth-s/<latest flashdepth-s checkpoint .pth> hybrid_configs.teacher_model_path=configs/flashdepth-l/<latest flashdepth-l checkpoint .pth> dataset.data_root=<path to data>
``` 

Check the `config.yaml` files in the `configs` folders for hyperparameters and logging.


## Timing / inference FPS
By default, we print out the wall time and FPS during inference. You can also run 
```
torchrun train.py --config-path configs/flashdepth inference=true eval.dummy_timing=true
```
to get the wall time over 100 frames (excluding warmup) with resolution 2044x1148. 
This is our console output on an A100 GPU:
```
INFO - shape: torch.Size([1, 105, 3, 1148, 2044]) 
INFO - wall time taken: 4.15; fps: 24.12; num frames: 100
```
As mentioned in the paper, we originally wrote [CUDA graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) but found that simply [compiling](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) the model provided similar performance.  


## Additional notes
### Temporal modules
We included an ablation study on a couple of temporal modules in our supplement. They include bi-directional Mamba (specifically, [Hydra](https://github.com/goombalab/hydra)), [xLSTM](https://github.com/NX-AI/xlstm), and an transformer-based RNN inspired by [CUT3R](https://github.com/CUT3R/CUT3R). Simply set the flags in `config.yaml` and install required packages if you want to use them (e.g. set `use_xlstm=true` but make sure `use_mamba` is still true).

### Placement of Mamba layers
In the paper we reported placing the mamba layers after the last DPT layer. In this repo we moved them to after the first DPT layer in FlashDepth (Full). See `mamba_in_dpt_layer` in `config.yaml`. We will update the paper later.


## References
Our code was modified and heavily borrowed from the following projects: <br>
[Depth Anything V2](https://depth-anything-v2.github.io/) <br>
[Mamba 2](https://github.com/state-spaces/mamba)

If you find our code or paper useful, please consider citing
```bibtex
@inproceedings{chou2025flashdepth,
  title     = {FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution},
  author    = {Chou, Gene and Xian, Wenqi and Yang, Guandao and Abdelfattah, Mohamed and Hariharan, Bharath and Snavely, Noah and Yu, Ning and Debevec, Paul},
  journal   = {The IEEE International Conference on Computer Vision (ICCV)},
  year      = {2025},
}
```
