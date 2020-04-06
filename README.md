# ciGAN2

Given the advances in deep learning in the generation of realistic images, the project aims to increase the pathology images of a mammography dataset synthetically, using Generation Antagonist Networks (GANs). 

Ref : ![mammo-GAN](https://github.com/ericwu09/mammo-cigan)

![](/images/demo.gif)

## Dependances
```python 
h5py==2.8.0
scipy==1.1.0
imageio==2.5.0
numpy==1.15.1
matplotlib==3.0.2
Keras==2.2.4
scikit_image==0.14.1
tensorflow==1.13.1
hickle==3.4.5
requests
```
## update code to tf2
```python 
tf_upgrade_v2 \
  --intree ./ \
  --outtree ./p2/ \
  --reportfile report.txt
``` 

# Download VGGModel and ciGAN2 pretrained
`python3 download_model.py`

Unzip models.zip in the same folder. The model must be in `./models`

# to validate the model

`python3 run.py --val`

# to synthesize one patch

`python3 run.py --syn id`

# to synthesize 8 patches

`python3 run.py --convert`
