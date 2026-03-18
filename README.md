
# Recommender Systems: Content-Based and Collaborative Filtering

## Overview
This project implements multiple recommender system techniques using the MovieLens dataset. It includes content-based filtering, collaborative filtering, matrix factorization (SVD), neural networks, and reinforcement learning approaches.

---

## Project Structure

├── recommender_system.ipynb    
├── README.md  
├── movies.csv  
├── ratings.csv  
├── images  
    ├── cbf_output.png  
    ├── user_profile_output.png  
    ├── cf_output.png  
    ├── svd_output.png  
    └── neural_network_output.png  

---

## Step-by-Step Workflow

### 1. Environment Setup
Install required libraries:

```python
!pip install numpy==1.26.4
!pip install pandas scikit-learn scipy tensorflow scikit-surprise
```

Restart kernel after installation.

---

### 2. Load Dataset

```python
import pandas as pd

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
```

---

### 3. Content-Based Filtering
- Extract genres
- Apply TF-IDF
- Compute cosine similarity
- Recommend similar movies

---

### 4. User Profile Recommendation
- Compute weighted average of movie vectors
- Use user ratings as weights

---

### 5. Collaborative Filtering

#### User-Based CF
- Compute user similarity
- Predict ratings

#### Item-Based CF
- Compute item similarity
- Predict ratings

---

### 6. Matrix Factorization (SVD)
- Create user-item matrix
- Apply SVD
- Reconstruct matrix
- Recommend movies

---

### 7. Neural Network Recommender
- Create user & movie features
- Build dense network
- Train using MSE loss

---

### 8. Reinforcement Learning
- Use ε-greedy strategy
- Update reward estimates

---

### 9. Evaluation Metrics
- RMSE
- Precision@K
- Recall@K

---

## Errors Faced and Solutions

### Error 1: SyntaxError in pip install
```
SyntaxError: invalid syntax
```
Fix:
Use:
```python
!pip install package_name
```

---

### Error 2: NumPy Compatibility Issue
```
ImportError: numpy.core.multiarray failed to import
```
Fix:
```python
!pip install numpy==1.26.4
```

---

### Error 3: No module named 'surprise'
Fix:
```python
!pip install scikit-surprise
```

---

### Error 4: Dataset Not Loading
Fix:
Ensure CSV files are in same directory or upload them properly.

---

## Results
- SVD RMSE ≈ 0.87
- Neural network shows decreasing loss
- CF predictions are consistent

---

## Author
Divyansh Rawat  
M25CSA009  
MTech AI  
IIT Jodhpur
