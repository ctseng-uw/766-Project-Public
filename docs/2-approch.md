# Approach & Implmentation

### 1. CycleGAN

### 2. DiscoGAN

### 3. DualGAN
DualGAN is a model purposed by Yi[1] etc. For the generator part, it adopts the U-shape net structure. This helps the model to share low-level information and keep the alignment of the image structures.

For the discriminator part, it employs the Markovian PatchGAN model. It keeps the independence between pixels distanced beyond a specific patch size and it is effective in capturing ocal high frequency.

All above characteristics of DualGAN contributes to a more stable output structure. However, too much constraints may also lead to the underfitting problem.

<p align="center">
  <img style={{width: 900}} src={require('./img/dualGAN.png').default} />
  <figcaption>The dualGAN model structure[1]</figcaption>
</p>



### 4. Our methods


# References
1. Yi, Zili & Zhang, Hao & Tan, Ping & Gong, Minglun. (2017). DualGAN: Unsupervised Dual Learning for Image-to-Image Translation. 2868-2876. 10.1109/ICCV.2017.310. 
