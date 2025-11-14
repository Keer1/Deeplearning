# **GAN Variants Implementation on CIFAR-10**

Implementation and comparison of three GAN architectures — **DCGAN**, **WGAN**, and **ACGAN** — trained on the CIFAR-10 dataset.  
This project evaluates generative quality, training stability, and class-conditioning performance across architectures.

---

## ** Models Performance**

This table summarizes the best FID scores achieved during training:

| **Model** | **Initial FID** | **Best FID** | **Final FID** |
|----------|------------------|--------------|----------------|
| **DCGAN** | 472.20 | **72.64** (epoch 48) | 76.82 |
| **WGAN** | 367.75 | **73.76** (epoch 28) | 80.76 |
| **ACGAN** | 379.91 | **162.53** (epoch 14) | 193.56 |

### **Key Insight**
- **DCGAN achieved the best overall score (72.64)**  
- **WGAN followed closely** with smooth and stable convergence  
- **ACGAN achieved strong classification accuracy** but significantly higher FID  

---

## ** Repository Structure**

Each model directory contains the following:

### **GeneratedImages/**
- Image grids produced at every 5 epoch  
- Final best-epoch samples  

### **RealImages/**
- Real CIFAR-10 image grids for comparison  

### **Metrics/**
- Generator & discriminator loss curves  
- Accuracy plots (ACGAN only)  
- FID progression for all epochs  

 

---

## **Training Environment**

Training was conducted on the **Palmetto Cluster**:

- **GPU:** NVIDIA V100  
- **GPU Memory:** 8 GB  
- **CPU:** 15 cores  
- **RAM:** 60 GB  


---

## ** Model Configuration**

### **Common Parameters**
- **Epochs:** 50  
- **Batch Size:** 128  
- **Learning Rate:** 0.0002  
- **Optimizer:** Adam (β1 = 0.5, β2 = 0.999)  
- **Image Size:** 64 × 64 × 3  
- **Latent Dimension:** 100  
- **Normalization:** Images scaled to **[-1, 1]**

---

## ** DCGAN**

### **Architecture Characteristics**
- Standard convolutional generator & discriminator  
- BatchNorm in generator  
- LeakyReLU activations  
- Tanh output  

### **DCGAN Final Results**
- **Best FID:** 72.64  
- **Final Generator Loss:** 3.8849  
- **Final Discriminator Loss:** 0.3566  
- **Discriminator Accuracy:** 94.64%  
- **Visual Quality:** Clear, stable shapes with strong realism  

---

## ** WGAN**

### **Special Parameters**
- **Critic Iterations:** 3  
- **Gradient Penalty:** λ = 10  
- **Wasserstein Loss** for improved stability  
- No sigmoid activation in critic  

### **WGAN Final Results**
- **Best FID:** 73.76  
- **Final Critic Loss:** –4.8228  
- **Final Generator Loss:** 75.2243  
- **Critic Accuracy:** 50% (expected for WGAN)  
- **Visual Quality:** Smooth textures, stable training, mild blurriness  

---

## **ACGAN**

### **Special Parameters**
- **Classes:** 10 (CIFAR-10)  
- Discriminator performs:  
  - Source prediction (real/fake)  
  - Class prediction (0–9)  

### **ACGAN Final Results**
- **Best FID:** 162.53  
- **Final D Loss:** 0.2587  
- **Final G Loss:** 6.5325  
- **Source Accuracy:** 98.85%  
- **Class Accuracy:** 95.24%  
- **Visual Quality:** Good class structure but noisy, less realistic images  

---

## **Results Summary**

### **Performance Overview**
- **DCGAN produced the best image quality and lowest FID**  
- **WGAN had the most stable training**, benefiting from Wasserstein distance and GP loss  
- **ACGAN excelled in classification** but struggled with image fidelity  

### **Final FID Values**
- **DCGAN:** 76.82  
- **WGAN:** 80.76  
- **ACGAN:** 193.56  

---

## ** Conclusion**

This project highlights the strengths and trade-offs across GAN architectures:

- **DCGAN** offers competitive quality, stable training, and the best FID.  
- **WGAN** reduces mode collapse and improves stability through Wasserstein loss and gradient penalty.  
- **ACGAN** performs excellent class-conditioned generation but compromises realism and FID scores.

For full visual outputs and per-epoch evolution, refer to each model’s corresponding results directory.

---
