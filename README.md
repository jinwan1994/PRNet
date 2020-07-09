# Progressive Residual Networks for Image Super-Resolution 
[[Applied Intelligence]](http://link.springer.com/article/10.1007/s10489-019-01548-8)
-------------
This repository is for PRNet introduced in the following paper


The code is based on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and tested on Ubuntu 18.04 environment (Python3.6, PyTorch_0.4.0, CUDA8.0, cuDNN5.1) with Titan X/1080Ti/Xp GPUs. .

![framework](/figs/Framework.png)

The architecture of our proposed Progressive Residual Networks (PRNet). The details about our proposed PRNet can be found in [our main paper](http://link.springer.com/article/10.1007/s10489-019-01548-8).

If you find our work useful in your research or publications, please consider citing:

```latex
@article{Jin2020Progressive,
  title={Progressive residual networks for image super-resolution},
  author={Jin Wan and Hui Yin and Ai-Xin Chong and Zhi-Hao Liu},
  journal={Applied Intelligence},
  number={7},
  year={2020},
}
```

## Contents
1. [Test](#test)
2. [Results](#results)
3. [Acknowlegements](#acknowledgements)

## Test

1. Clone this repository:

   ```shell
   git clone https://github.com/jinwan1994/PRNet.git
   ```

2. Download our trained models from [BaiduYun](https://pan.baidu.com/s/13QxG0S4ErCvY81q2x6io5A)(code:en3h), place the models to `./models`. We have provided three small models (LFFN_x2_B4M4_depth_div2k/LFFN_x3_B4M4_depth_div2k/LFFN_x4_B4M4_depth_div2k) and the corresponding results in this repository.

3. Place SR benchmark (Set5, bsd100, Urban100 and Manga109) or other images to `./data/test_data/*`.

4. You can edit `./helper/args.py` according to your personal situation.

5. Then, run **following command** for evaluation:
   ```shell
   python evaluate.py
   ```

6. Finally, SR results and PSNR/SSIM values for test data are saved to `./models/*`. (PSNR/SSIM values in our paper are obtained using matlab)

## Results

#### Quantitative Results

![](figs/benchmark.png)

Benchmark SISR results. Average PSNR/SSIM for scale factor x2, x3 and x4 on datasets Set5, Manga109, bsd100 and Urban100.

#### Visual Results

![](figs/visual_compare.png)

Visual comparison for x3 SR on â€œimg013â€? â€œimg062â€? â€œimg085â€from the Urban100 dataset.

## Acknowledgements

- Thank [DCSCN](https://github.com/jiny2001/dcscn-super-resolution). Our code structure is derived from it. 
- Thank [BasicSR](https://github.com/xinntao/BasicSR). They provide many useful codes which facilitate our work.

