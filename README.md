# 1 环境配置

```shell
conda create -n udis python=3.8
conda activate udis
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt  # 指定numpy版本，不能超过1.20
```

# 2 Warp

## 2.1 训练

设置 `Warp/Codes/train.py` 中的训练数据集路径
  
```shell
python train.py
```

## 2.2 测试

### 2.2.1 计算 PSNR/SSIM

设置 `Warp/Codes/test.py` 中的测试数据集路径
```shell
python test.py --gpu 0
```

### 2.2.2 Generate the warped images and corresponding masks

设置 `Warp/Codes/test_output.py` 中的训练/测试数据集路径
```shell
python test_output.py --gpu 0
```
变形后的图像、变形后图像掩码、平均融合结果均会保存在数据集路径下

### 2.2.3 Test on other datasets

当在不同场景和分辨率的其他数据集上进行测试时，应用迭代扭曲适应来微调预训练模型，获得更好的对齐性能。设置 `Warp/Codes/test_other.py` 中的 `path/img1_name` 和 `path/img2_name`，默认情况下，`img1` 和 `img2` 都放在 `path` 下
```shell
python test_other.py --gpu 0
```

微调前后的结果保存在 `path` 下


------

# <p align="center">Parallax-Tolerant Unsupervised Deep Image Stitching (UDIS++ [paper](https://arxiv.org/abs/2302.08207))</p>
<p align="center">Lang Nie*, Chunyu Lin*, Kang Liao*, Shuaicheng Liu`, Yao Zhao*</p>
<p align="center">* Institute of Information Science, Beijing Jiaotong University</p>
<p align="center">` School of Information and Communication Engineering, University of Electronic Science and Technology of China</p>

![image](fig1.png)

## Dataset (UDIS-D)
We use the UDIS-D dataset to train and evaluate our method. Please refer to [UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching) for more details about this dataset.


## Code
#### Requirement
* numpy 1.19.5
* pytorch 1.7.1
* scikit-image 0.15.0
* tensorboard 2.9.0

We implement this work with Ubuntu, 3090Ti, and CUDA11. Refer to [environment.yml](environment.yml) for more details.

#### How to run it
Similar to UDIS, we also implement this solution in two stages:
* Stage 1 (unsupervised warp): please refer to  [Warp/readme.md](https://github.com/nie-lang/UDIS2/blob/main/Warp/readme.md).
* Stage 2 (unsupervised composition): please refer to [Composition/readme.md](https://github.com/nie-lang/UDIS2/blob/main/Composition/readme.md).



## Meta
If you have any questions about this project, please feel free to drop me an email.

NIE Lang -- nielang@bjtu.edu.cn
```
@inproceedings{nie2023parallax,
  title={Parallax-Tolerant Unsupervised Deep Image Stitching},
  author={Nie, Lang and Lin, Chunyu and Liao, Kang and Liu, Shuaicheng and Zhao, Yao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7399--7408},
  year={2023}
}
```


## References
[1] L. Nie, C. Lin, K. Liao, M. Liu, and Y. Zhao, “A view-free image stitching network based on global homography,” Journal of Visual Communication and Image Representation, p. 102950, 2020.  
[2] L. Nie, C. Lin, K. Liao, and Y. Zhao. Learning edge-preserved image stitching from multi-scale deep homography[J]. Neurocomputing, 2022, 491: 533-543.   
[3] L. Nie, C. Lin, K. Liao, S. Liu, and Y. Zhao. Unsupervised deep image stitching: Reconstructing stitched features to images[J]. IEEE Transactions on Image Processing, 2021, 30: 6184-6197.   
[4] L. Nie, C. Lin, K. Liao, S. Liu, and Y. Zhao. Deep rectangling for image stitching: a learning baseline[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 5740-5748.   
