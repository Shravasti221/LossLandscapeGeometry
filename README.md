
Install the following
```
!pip install loss-landscapes
!pip install matplotlib
!pip install --upgrade git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings
```

# Loss Landscape Geometry & Optimization Dynamics

## Architecture Comparison and Optimizer Sweep Study

---

## **1. Introduction**

Understanding why deep neural networks generalize well despite their highly non-convex loss surfaces remains one of the central questions in deep learning theory. Empirically, Stochastic Gradient Descent (SGD) converges reliably to solutions that perform well on unseen data, even though the loss surface contains many saddle points and numerous minima of varying quality.

This project investigates the geometric structure of neural network loss landscapes and how architectural choices and optimization algorithms shape these surfaces. Specifically, we explore:

### **Key Questions**

1. **Architecture Impact:**
   How do different neural architectures (MLPs vs. CNNs) affect the structure of the loss landscape?

2. **Width Scaling:**
   How does increasing model capacity influence curvature, smoothness, and optimization speed?

3. **Geometric–Performance Correlation:**
   Can landscape geometry predict optimization difficulty and generalization quality?

Our experiments combine random-direction probing, 1D and 2D landscape visualization, and Hessian eigenvalue analysis to develop a multi-scale understanding of curvature and basin structure.

---

## **2. Empirical Results: Architecture & Landscape Geometry**

We trained multiple MLP and CNN architectures on MNIST using SGD (learning rate 0.01, momentum 0.9). For each model, we measured:

* training and test loss,
* training speed,
* Hessian curvature at convergence,
* qualitative landscape smoothness.

All landscape probing used **filter-normalized perturbations** to ensure comparable geometric scales across architectures.

---

### **2.1 Training Performance Summary**

CNNs converge significantly faster and achieve better minima than MLPs of comparable width.
Example observations:

* Narrow MLP (width 32) requires 24 epochs to reach ~97% accuracy.
* Wide MLP (width 128) reaches ~98% accuracy in only 12 epochs.
* CNNs are dramatically more efficient:

  * even CNN-16 reaches ~98.8% accuracy,
  * CNN-64 achieves ~99.1% accuracy in just 7 epochs.

This suggests the convolutional inductive bias strongly influences the landscape, making minima both easier to reach and more favorable.

---

### **2.2 Landscape Structure**

**MLPs:**
As width increases:

* 1D directions become smoother,
* the basin around the minimum becomes flatter,
* curvature decreases,
* 2D valleys transition from elongated and narrow to more circular and symmetric.

**CNNs:**
Even the narrowest CNNs exhibit:

* extremely wide basins,
* smooth and nearly convex regions,
* a strong architectural bias toward global flatness.

Interestingly, CNNs can appear locally sharp (in Hessian spectral terms) but globally flat when visualized over larger perturbations—revealing a separation between local curvature and global basin geometry.

---

### **2.3 Hessian Spectral Trends**

Dominant eigenvalues for MLPs generally **fall** as width increases.
CNNs, however, sometimes show **larger local curvature**, but their global geometry is much smoother.

This reconciles conflicting ideas in the literature:

* “Flat minima generalize better” (Hochreiter & Schmidhuber, 1997)
* “Sharp minima can also generalize” (Dinh et al., 2017)

CNNs demonstrate that local sharpness and global flatness can coexist.

---

## **3. Inferences from Architecture Experiments**

### **(1) How architecture affects geometry**

CNNs create smoother, more connected landscapes than MLPs, regardless of raw parameter count.
Weight sharing and translation equivariance reshape the surface, producing broad basins with fewer pathological curvature directions.

### **(2) Geometry vs trainability/generalization**

* **Global flatness** strongly correlates with fast convergence and high accuracy.
* **Local sharpness** correlates negatively in MLPs, but not reliably in CNNs.
  This indicates that global structure matters more than pointwise Hessian values.

### **(3) Predicting optimization difficulty**

A single 2D contour plot qualitatively predicts training time:

* Rugged/elongated → slow
* Wide/smooth → fast

This holds across all six models tested.

### **(4) Why SGD finds good minima**

SGD is guided into broad, connected low-loss plateaus that occupy large volumes in parameter space. These are easy to reach and robust to noise, explaining its surprisingly reliable generalization.

---

## **4. Optimizer Sweep: How SGD, Momentum, and Adam Shape Geometry**

Phase 2 isolates the effect of optimization algorithms by training each architecture for exactly five epochs under:

* SGD
* SGD + Momentum
* Adam

After training, we compute:

* training curves,
* 1D and 2D landscape slices,
* Hessian eigenvalues.

### **4.1 Results for MLPs**

**SGD:**

* slowest convergence
* sharpest minima
* highly anisotropic basins

**Momentum-SGD:**

* smoothest 1D curves
* flattest minima
* best accuracy
* nearly symmetric quadratic basins

**Adam:**

* flatter than SGD
* slightly sharper than momentum
* hybrid geometry

### **4.2 Results for CNNs**

CNNs amplify optimizer differences:

**SGD:**

* extremely sharp minima
* high curvature (~118 eigenvalue)

**Momentum-SGD:**

* curvature collapses from ~118 to ~17
* wide, stable basins
* dramatic geometric improvement

**Adam:**

* similar to momentum, moderately flat
* smoother than SGD but less uniform than momentum

### **4.3 Key Takeaways**

* SGD gravitates to sharp minima.
* Momentum finds the flattest minima.
* Adam is intermediate.
* CNNs exaggerate these geometric effects.

These findings unify theoretical claims:

* momentum acts as a low-pass filter,
* adaptive optimizers flatten gradients directionally,
* SGD alone is sensitive to parameter scaling and curvature.

---

## **5. SGD Finds Wide, Connected, Flat Minima**

To understand why SGD generalizes well, we analyze:

* linear mode connectivity,
* PCA directions of SGD trajectories,
* perturbation-based flatness metrics.

### **5.1 Linear Mode Connectivity**

Models trained from different initializations lie in the **same connected valley**:

* no spikes or discontinuities along the interpolation path,
* low relative barriers,
* smooth transitions between minima.

This confirms the presence of **large contiguous manifolds of good solutions**—a hallmark of over-parameterized models.

### **5.2 Optimization Trajectory Geometry**

When projecting SGD trajectory into its PCA basis:

* the dominant direction forms a perfect parabola,
* loss variation is extremely small (~$10^{-3}$),
* indicating convergence to the center of a broad valley.

### **5.3 Perturbation-Based Flatness**

Despite CNNs showing large Hessian eigenvalues:

* random perturbations reveal extremely wide basins,
* confirming that Hessian-only metrics can be misleading.

Combined, the evidence shows SGD does not settle into isolated sharp minima, but rather into wide, stable manifolds.

---

## **6. Conclusion**

This study establishes a coherent understanding of how architecture and optimization jointly shape the geometry of loss landscapes in deep neural networks. Our major conclusions:

1. **Architecture is the dominant factor in global geometry.**
   CNNs naturally create smoother, more connected basins than MLPs.

2. **Width enhances flatness.**
   Wide MLPs exhibit much lower curvature and faster convergence.

3. **Optimizer choice influences local geometry.**
   Momentum flattens basins; Adam partially flattens; SGD remains sharp.

4. **SGD finds generalizable minima because global flatness matters more than local sharpness.**

5. **Landscape visualization is indispensable.**
   Hessian values alone cannot capture global basin width or connectivity.

This integration of curvature analysis, trajectory geometry, and optimizer dynamics provides a rigorous empirical basis for understanding deep learning optimization.

---

## **7. References**

* Jacot et al., *Neural Tangent Kernel*, NeurIPS 2018
* Goodfellow et al., *Deep Learning*, MIT Press
* Kawaguchi, *Deep Learning Without Poor Local Minima*, NeurIPS 2017
* Li et al., *Visualizing the Loss Landscape of Neural Nets*, NeurIPS 2018

---
