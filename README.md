# Recommender Systems: Content-Based and Collaborative Filtering

This project implements and compares multiple recommender system techniques using the **MovieLens dataset**.  
The objective is to explore how different recommendation algorithms perform when predicting user preferences for movies.

The assignment was implemented in a **Jupyter Notebook** and includes multiple recommendation approaches such as content-based filtering, collaborative filtering, matrix factorization, neural networks, and reinforcement learning.

---

# Project Structure

.
├── recommender_system.ipynb  
├── report.pdf  
├── README.md  
├── images  
│   ├── cbf_output.png  
│   ├── user_profile_output.png  
│   ├── cf_output.png  
│   ├── svd_output.png  
│   └── neural_network_output.png  

---

# Dataset

The project uses the **MovieLens Small Dataset**.

Files used:

- `movies.csv` – contains movie titles and genres
- `ratings.csv` – contains user ratings for movies

Dataset statistics:

| Feature | Value |
|------|------|
| Users | 610 |
| Movies | 9742 |
| Ratings | 100,836 |
| Rating scale | 0.5 – 5 |

Dataset source:
https://grouplens.org/datasets/movielens/

---

# Implemented Recommendation Methods

### 1. Content-Based Filtering
- Uses movie genres as textual features
- Converts genres to vectors using **TF-IDF**
- Computes similarity using **cosine similarity**

### 2. User Profile Based Recommendation
Builds a user preference vector using weighted movie features.

Formula:

Pu = Σ(ru,m * fm) / Σ(ru,m)

### 3. User-Based Collaborative Filtering
- Finds users with similar rating patterns
- Predicts ratings using weighted average of similar users

### 4. Item-Based Collaborative Filtering
- Computes similarity between movies
- Predicts ratings using ratings of similar movies

### 5. Matrix Factorization (SVD)
R ≈ U Σ Vᵀ

Used to generate movie recommendations.

### 6. Neural Network Recommender
- Two input branches for user and movie features
- Dense layers generate embeddings
- Final dense layer predicts rating

### 7. Reinforcement Learning Recommender
Uses an ε-greedy strategy to balance exploration and exploitation.

---

# Evaluation Metrics

### RMSE
RMSE = sqrt(1/N Σ(r_true − r_pred)²)

### Precision@K
Measures how many recommended movies are relevant.

### Recall@K
Measures how many relevant movies were recommended.

---

# Results Summary

| Model | RMSE |
|------|------|
| SVD | 0.87 |

SVD produced the best prediction performance among the implemented methods.

---

# Requirements

Install the following libraries:

pip install pandas  
pip install numpy  
pip install scikit-learn  
pip install scipy  
pip install scikit-surprise  
pip install tensorflow  

---

# How to Run

1. Clone the repository

git clone https://github.com/your-repo-name.git

2. Navigate to the project directory

cd recommender-system

3. Launch Jupyter Notebook

jupyter notebook

4. Run all cells in the notebook.

---

# Common Errors Encountered

### Error 1: Surprise Library Import Error
ImportError: numpy.core.multiarray failed to import

Solution:
pip install numpy==1.26.4  
pip install scikit-surprise  

### Error 2: No Module Named 'surprise'
ModuleNotFoundError: No module named 'surprise'

Solution:
pip install scikit-surprise

### Error 3: CSV File Not Found
FileNotFoundError: movies.csv

Solution:
Ensure `movies.csv` and `ratings.csv` are placed in the same directory as the notebook.

### Error 4: Colab Runtime Compatibility Issue
NumPy version incompatibility with Surprise.

Solution:
pip install numpy==1.26.4  
pip install scikit-surprise  

---

# Assumptions

- Movie genres are used as primary content features
- Ratings ≥ 4 are treated as positive feedback
- Missing ratings are filled with zero
- Only the MovieLens small dataset is used

---

# Author

Divyansh Rawat  
M25CSA009  
MTech Artificial Intelligence  
Indian Institute of Technology Jodhpur
