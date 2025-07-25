# Movie-Rating-Prediction-System

A machine learning system that predicts movie ratings (1-10) using multiple algorithms and movie features.

##Features

- **6 ML Algorithms**: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, Decision Tree
- **Smart Data Processing**: Handles numerical, categorical, and text data
- **Auto Model Selection**: Picks best performing model
- **Visualizations**: EDA charts and model comparisons
- **Easy Predictions**: Simple API for new movies

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Quick Start

```python
from movie_predictor import MovieRatingPredictor

# Initialize and train
predictor = MovieRatingPredictor()
df = predictor.load_and_prepare_data()  # Uses sample data
X, y = predictor.preprocess_data(df)
X_test, y_test = predictor.train_models(X, y)
predictor.evaluate_models(X_test, y_test)

# Make prediction
new_movie = {
    'budget': 75000000,
    'runtime': 135,
    'year': 2023,
    'popularity': 8.5,
    'vote_count': 2500,
    'genre': 'Action',
    'director': 'Christopher Nolan',
    'description': 'Epic sci-fi thriller with stunning visuals'
}

rating = predictor.predict_rating(new_movie)
print(f"Predicted Rating: {rating:.1f}/10")
```

## Dataset Format

Required CSV columns:
- `genre`, `director`, `budget`, `runtime`, `year`, `popularity`, `vote_count`, `description`, `rating`

Example:
```csv
title,genre,director,budget,runtime,year,popularity,vote_count,description,rating
Inception,Sci-Fi,Christopher Nolan,160000000,148,2010,29.1,14075,"A mind-bending thriller",8.8
```

## Models & Metrics

**Algorithms**: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, Decision Tree

**Evaluation**: MSE, MAE, R² Score, Cross-validation

**Features**: Budget, runtime, year, popularity, votes, genre, director, TF-IDF text

## Example Output

```
=== MODEL EVALUATION ===
Model                 R²       MAE    
Random Forest         0.8734   0.3421 
Gradient Boosting     0.8655   0.3567 
Ridge Regression      0.8561   0.3789 

Best Model: Random Forest

Predicted rating for new movie: 7.85/10
```

## Custom Dataset

```python
# Use your own data
df = predictor.load_and_prepare_data('your_movie_data.csv')
```

## Troubleshooting

- Ensure CSV has required columns
- Check data types (numeric for budget, runtime, etc.)
- Install all dependencies
- Use UTF-8 encoding for text data

**Quick 5-line workflow**: Load data → Preprocess → Train → Evaluate → Predict 
