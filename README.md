# NeRF-Reproduce

This project is a part of CS492(H) Machine Learning for 3D Data course at KAIST.
Note that the code for raw data preprocessing (i.e., converting a file to a NumPy array), sample video exporting, and training phase logging is borrowed from an open repository called [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch).

## Installation

```
git clone https://github.com/Mephisto405/NeRF-Reproduce.git
cd nerf-pytorch
pip install -r requirements.txt
```

## How To Run?

### Training from Scratch

We refer the [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) repository to train a model from scratch.
For the review procedure, I recommend you to use the pretrained weights and data that we provide in this repository directly:

### Testing with Pretrained Weights

The following commands produce 200 novel-view images (i.e., test set) in the Lego scene and the mean PSNR, SSIM, and LPIPS error:

```
python run_nerf.py --config configs/lego_complete_HR.txt
```

If a memory-related error appears after processing the 200 images, remove the comments on lines 663-664 of run_nerf.py and run the same command again. Or you can reduce the chunk size in the configs/lego-or-fern_complete_HR.txt. See the comments in the files.

Also, we note that the errors can be slightly different with the numbers we present in the report, due to the stochatic nature of volumetric rendering (we refer the Equation 1 in the report).

The following command produce 3 novel-view images in the Fern scene and metrics as above:

```
python run_nerf.py --config configs/fern_complete_HR.txt
```

## Citation
The original work:
```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

The PyTorch re-implementation that we utilize:
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```
