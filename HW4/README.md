# CIFAR-10 GAN Experiments

This project explores three different GAN designs — **DCGAN**, **WGAN**, and **ACGAN** — and compares how well they can generate CIFAR-10 images.  
All models were trained for 50 epochs using identical hyperparameters.

---

## Overall Results

| Model | Best FID | Notes |
|-------|----------|-------|
| **DCGAN** | **72.64** | Sharpest samples and strongest performance |
| **WGAN** | 73.76 | Very stable training, smooth convergence |
| **ACGAN** | 162.53 | High class accuracy but weaker image quality |

**DCGAN produced the best results**, while **WGAN was a close second** with more stable behavior.  
ACGAN successfully learned class labels but struggled to maintain image fidelity.

---

##  Folder Layout

Each model folder contains:
- **GeneratedImages** – images created during training  
- **RealImages** – CIFAR-10 reference grids  
- **Metrics** – FID curves, loss plots, accuracy plots  
  

---

## ⚙ Training Configuration

- **Epochs:** 50  
- **Batch Size:** 128  
- **Latent Size:** 100  
- **Optimizer:** Adam (lr=0.0002, β1=0.5)  
- **Image Size:** 64×64 RGB  
- **Normalization:** [-1, 1]  
- **Environment:** Palmetto cluster (V100 GPU)

---

##  Model Highlights

###  DCGAN
- Best FID: **72.64**  
- Final D Accuracy: **94.64%**  
- Produces the cleanest and most consistent images

###  WGAN
- Best FID: **73.76**  
- Critic behaves very steadily as expected from Wasserstein training  
- Images improve quickly and remain stable

###  ACGAN
- Best FID: **162.53**  
- Class Accuracy: **95.24%**  
- Good label consistency but lower realism

---

##  Summary

- **DCGAN** → Best balance of quality and stability  
- **WGAN** → Most reliable training dynamics  
- **ACGAN** → Strong classifier but noisier samples  

For visual examples and full training logs, check the folders for each model.
