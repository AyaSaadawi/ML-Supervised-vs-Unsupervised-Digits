# **Unified Comparative Framework for Clustering and Classification of Handwritten Digits with KMeans, GMM, kNN, and CNN**  
### *A Comparative Study of KMeans, GMM, kNN, and CNN on the Digits Image Dataset*  
*(Includes a high-quality video visualization of all four models in action)*  

---

## **Project Overview**

This project was developed as part of my exploration of machine learning and computer vision methodologies.  
It provides an **educational, end-to-end study** of classical ML, unsupervised clustering, and deep learning techniques using the well-known **Digits Image Dataset** (8×8 grayscale digit images).

It combines:

- **Unsupervised Learning**
  - **K-Means Clustering**
  - **Gaussian Mixture Models (GMM)**

- **Supervised Learning**
  - **k-Nearest Neighbors (kNN)**

- **Deep Learning**
  - **Convolutional Neural Networks (CNN)**

All models are trained, visualized, and compared within a unified pipeline.  
A **custom animation video** visualizes how each algorithm progressively organizes and classifies digit samples in a 2D PCA space.

---

## **Visualization Video Included**

The project includes a custom animation that simultaneously shows:

- **KMeans:** iterative centroid movement  
- **GMM:** evolving Gaussian ellipses and covariance structures  
- **kNN:** prediction patterns and class-color distributions  
- **CNN:** deep-learning predictions projected into 2D  

The visualization is designed as a **teaching aid** to make algorithmic mechanisms intuitive and visually clear.

---

## **Educational Purpose**

This project focuses heavily on conceptual clarity and interpretability.

### **Mathematical Foundations**
The repository includes explanations of:

- KMeans objective function & iterative optimization  
- Expectation–Maximization (E–M) for GMM  
- kNN decision boundaries & distance metrics  
- Convolutional layers, pooling, and dense blocks in CNNs  
- PCA for dimensionality reduction  
- Cluster evaluation metrics: Silhouette, Davies–Bouldin, Calinski–Harabasz  

### **Algorithmic Mechanisms**
Step-by-step demonstrations cover:

- Centroid updates in KMeans  
- Covariance-driven cluster shapes in GMM  
- Example-based decisions in kNN  
- Feature extraction and non-linear classification in CNNs  

---

## **Comparative Analysis**

The notebook includes a thorough evaluation of all models:

### **Unsupervised Models: KMeans & GMM**
- Silhouette Score  
- Davies–Bouldin Index  
- Calinski–Harabasz Score  
- AIC / BIC (for GMM)  
- Cluster compactness & separation  

### **Supervised Models: kNN & CNN**
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion matrices  

### **Runtime Complexity Benchmark**
Empirical timing analysis was conducted for:

- KMeans  
- GMM  
- kNN  
- CNN  

to assess computational cost as sample size increases.

---

## **Key Takeaways**

This project helps demonstrate:

- Differences between **unsupervised**, **supervised**, and **deep learning** approaches  
- How GMM captures richer distribution structures than KMeans  
- Performance contrasts between classical ML and CNNs on image data  
- Strength of PCA for visual interpretability  
- How to create algorithm animations that reveal internal learning dynamics  

---

## **Technologies Used**

- Python  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-Learn (KMeans, GMM, PCA, kNN)  
- TensorFlow / Keras (CNN)  
- ImageIO (video generation)  

---

## **Authorship**

All analysis, visualizations, animations, and educational content were authored by **Aya Saadaoui**.  
This project is part of a series focused on building intuitive visual tools for understanding machine learning and computer vision.

---

## **Support the Project**

If you find this work helpful, please consider **starring the repository** and sharing it with students or ML enthusiasts.
