<p align="center">

  <h2 align="center">StyleSculptor: Zero-Shot Style-Controllable 3D Asset Generation with Texture-Geometry Dual Guidance</h2>
  <p align="center">
    <a href="https://quzefan.github.io/"><strong>Zefan Qu</strong></a>
    ·
    <a href="https://zhenwwang.github.io/"><strong>Zhenwei Wang</strong></a>
    ·
    <a href="https://www.whyy.site/"><strong>Haoyuan Wang</strong></a>
    ·
    <a href="https://kkbless.github.io/"><strong>Ke Xu</strong></a>
    ·
    <a href="https://rfidblog.org.uk/"><strong>Gerhard Hancke</strong></a>
    ·
    <a href="https://www.cs.cityu.edu.hk/~rynson/"><strong>Rynson W.H. Lau</strong></a>
    <br>
    <br>
        <a href="https://arxiv.org/abs/2509.13301"><img src='https://img.shields.io/badge/arXiv-AntiReference' alt='PDF - todo'></a>
        <a href="https://StyleSculptor.github.io"><img src="https://img.shields.io/badge/Project%20Page-StyleSculptor-blue" alt="Project Page"></a>
        <br>
    <b>City University of Hong Kong</b>
  </p>

## 📢 News
* **[Aug.10.2025]** StyleSculptor is accepted to SIGGRAPH Asia 2025 🍀. The code is still being organized. Stay tuned！
* **[Oct.15.2025]** We release the inference code of dual-style guidance generation. 

<!-- Installation -->
## 📦 Installation
Our code is highly built on the <a href="https://github.com/microsoft/TRELLIS/"><strong>TRELLIS</strong></a> Repo, you can follow their offical guidance and find the solutions of the installation problems.

### Prerequisites
- **System**: The code is currently tested only on **Linux**.  
- **Hardware**: An NVIDIA GPU with at least 16GB of memory is necessary. The inference time of each case is about 2 minutes on a NVIDIA RTX 4090.
- **Software**:   
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain submodules.  
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is recommended for managing dependencies.  
  - Python version 3.8 or higher is required. 

### Installation Steps
1. Clone the repo:
    ```sh
    git clone https://github.com/quzefan/StyleSculptor
    cd StyleSculptor
    ```

2. Install the dependencies:
Create a new conda environment named `stylesculptor` and install the dependencies:
    ```sh
    . ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
    ```

<!-- Pretrained Models -->
## 🤖 Pretrained Models
We don't modify the pretrained models of TRELLIS. You can directly load the models with their repository names in the code:
```python
TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
```

You can also load the model from local by downloading the checkpoint (TRELLIS-image-large) from <a href="https://github.com/microsoft/TRELLIS/"><strong>TRELLIS</strong></a> Repo, then change the code like: 
```python
TrellisImageTo3DPipeline.from_pretrained("/path/to/TRELLIS-image-large")
 ```
 For more information, please refer to <a href="https://github.com/microsoft/TRELLIS/"><strong>TRELLIS</strong></a>.

 <!-- Inference -->
## 💡 Inference
### Data Preparation

 The edge map of each style images should be provided. The format can be referred in ```./asset/style_image_edge folder```.

 We use <a href="https://github.com/microsoft/TRELLIS/"><strong>PidiNet</strong></a> to generate the edge map for style images.

 ⭐⭐Make sure there is **main object** in the content and style images. If not, please turn off the **rembg** operation in the data preprocessing stage.

### Quick Start
```python
python example.py --cnt /path/cnt_image --sty /path/sty_image --sty_edge /path/sty_edge_image --intensity intensity_value
```

Parameters in the command:
- `cnt`: Content image(s). Image path / Multi-view images Folder path. 
- `sty`: Style image(s). Image path / Multi-view images Folder path. 
- `sty_edge`: Edge maps of all input style images.
- `intensity`: The style guidance intensity. Valid value: 0 (No Guidance) - 5 (Full Guidance).

### Example
```python
python example.py --cnt ./asset/content_multi_image/character --sty ./asset/style_image/groot_010.png --sty_edge ./asset/style_image_edge/groot_010.png --intensity 2
```

<!-- Citation -->
## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{qu2025stylesculptor,
  title={StyleSculptor: Zero-Shot Style-Controllable 3D Asset Generation with Texture-Geometry Dual Guidance},
  author={Qu, Zefan and Wang, Zhenwei and Wang, Haoyuan and Xu, Ke and Hancke, Gerhard and Lau, Rynson WH},
  journal={arXiv preprint arXiv:2509.13301},
  year={2025}
}
```

## Acknowledgements
Our codebase builds on [TRELLIS](https://github.com/microsoft/TRELLIS/).
Thanks the authors for sharing their awesome codebases! 