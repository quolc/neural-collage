
# Neural Collage
### [arXiv preprint](https://arxiv.org/abs/1811.10153)

Chainer implementation of our novel CNN-based image editing method that allows the user to change the semantic information of an image over a user-specified region:

[Collaging on Internal Representations: An Intuitive Approach for Semantic Transfiguration](https://arxiv.org/abs/1811.10153)

[Ryohei Suzuki](http://tealang.info/)<sup>1,2*</sup>, [Masanori Koyama](http://ishiilab.jp/member/koyama-m/)<sup>2</sup>, [Takeru Miyato](https://takerum.github.io/)<sup>2</sup>, Taizan Yonetsuji<sup>2</sup>  
 <sup>1</sup>The University of Tokyo,
 <sup>2</sup>Preferred Networks, Inc.,<br/>
 <sup>*</sup>This work was done when the author was at Preferred Networks, Inc.<br/>
 arXiv:1811.10153
 
 
## Collage-based image synthesis

### Spatial class-translation

### Semantic transplantation

### Spatial class + semantic transfiguration

### Editing existing images

## Setup
 
### Install required python libraries:
```bash
pip install -r requirements.txt
```

### Download ImageNet dataset (optional):
If you want to train the networks using ImageNet dataset, please download ILSVRC2012 dataset from http://image-net.org/download-images

### Preprocess dataset (optional):
```bash
cd datasets
IMAGENET_TRAIN_DIR=/path/to/imagenet/train/ # path to the parent directory of category directories named "n0*******".
PREPROCESSED_DATA_DIR=/path/to/save_dir/
bash preprocess.sh $IMAGENET_TRAIN_DIR $PREPROCESSED_DATA_DIR
# Make the list of image-label pairs for all images (1000 categories, 1281167 images).
python imagenet.py $PREPROCESSED_DATA_DIR
# Make the list of image-label pairs for dog and cat images (143 categories, 180373 images). 
python imagenet_dog_and_cat.py $PREPROCESSED_DATA_DIR
```

### Pre-trained models

Please download models from [this link](http://example.com/).

 
## Web-based demos

### Spatial class-translation
 
<p align='center'>  
  <img src='images/demo_class_translation.gif' width='640'/>
</p>

```bash
# launch server on localhost:5000
python demo_spatial_translation.py \
--config ./configs/sn_projection_dog_and_cat_256_scbn.yml \
--gen_model ./sn_projection_dog_and_cat_256/ResNetGenerator_450000.npz \
--gpu 0
```
 
### Semantic transplantation
 
<p align='center'>  
  <img src='images/demo_semantic_transplantation.gif' width='640'/>
</p>

```bash
# launch server on localhost:5000
python demo_feature_blending.py \
--config ./configs/sn_projection_dog_and_cat_256_scbn.yml \
--gen_model ./sn_projection_dog_and_cat_256/ResNetGenerator_450000.npz \
--gpu 0
```
 
## CoLab examples
 
(under preparation)

## Running individual scripts

### Spatial class-translation

```bash
python
```

### Feature blending with spatial class-translation

```bash
python
```

### Manifold projection

```bash
python
```

## Training models

### Generator + Discriminator (GAN)

Please see [snGAN-projection](https://github.com/pfnet-research/sngan_projection) for the detailed information.

```bash
python
```
 
### Encoder

```bash
python
```

### Auxiliary Network

```bash
python
```
