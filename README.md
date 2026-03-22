# Anime Success Predictor

A machine learning project that predicts whether an anime will become **highly rated (Score ≥ 8)** using metadata from the **MyAnimeList dataset**.

The model uses **logistic regression implemented from scratch using NumPy** and analyzes features such as popularity, engagement metrics, genres, and production details.

---

# Problem Statement

Anime ratings vary widely depending on popularity, audience engagement, genre, and production factors.

The goal of this project is to **predict whether an anime will be successful**, defined as:

```
success = 1  if score >= 8
success = 0  if score < 8
```

This converts the problem into a **binary classification task**.

---

# Dataset

Dataset source: **MyAnimeList Anime Dataset (Kaggle)**

The dataset contains metadata for thousands of anime titles including:

* Score
* Episodes
* Members
* Favorites
* Genres
* Type (TV, Movie, OVA, etc.)
* Source (Manga, Light Novel, Original, etc.)
* Studios
* Popularity
* Rank
* Scored By

---

# Exploratory Data Analysis (EDA)

Several visualizations were used to understand the dataset.

### Score Distribution

Most anime fall within the **5 – 7.5 score range**, while only a small percentage achieve scores above **8**.

### Anime Type Distribution

Most anime are **TV series**, followed by:

* OVA
* Movies
* Specials

### Episode Distribution

A surprisingly large number of anime have **only 1 episode**, which includes:

* Movies
* Specials
* Short productions

### Episodes vs Score

Anime with **12–24 episodes** tend to have higher average scores, as many popular seasonal anime fall in this range.

### Genre Distribution

Comedy appears very frequently across the dataset.
Since many anime have **multiple genres**, this distribution reflects tagging patterns as well.

### Popularity vs Score

Anime with more **members (viewers)** generally tend to receive more ratings and slightly higher scores.

### Studio Analysis

Older studios tend to have **larger catalogs of anime**.

When filtering studios with **at least 10 productions**, interesting differences in **average studio ratings** appear.

### Correlation Analysis

Key insights from the correlation heatmap:

* **Score and Rank** have a strong negative correlation (-0.98) because rankings are derived from ratings.
* **Members and Scored By** have an extremely strong positive correlation (0.99).
* **Members and Favorites** also show strong correlation, indicating viewer engagement.
* **Episodes** have very weak correlation with score, suggesting anime length alone does not determine quality.

---

# Feature Engineering

Several features were engineered to improve the predictive power of the model.

### Log Popularity Feature

Members can range from hundreds to millions, so we apply a logarithmic transformation:

```
members_log = log(members + 1)
```

This stabilizes the scale.

### Engagement Ratio

A useful feature representing audience enthusiasm:

```
fav/members = favorites / members
```

This measures how many viewers liked the anime enough to favorite it.

### Encoded Features

Categorical variables were converted to numerical features using **one-hot encoding**:

* Anime Type
* Source Material
* Genres

---

# Model

The model used is **Logistic Regression implemented from scratch using NumPy**.

Logistic regression estimates the probability:

```
P(success | features)
```

### Sigmoid Function

The sigmoid function converts linear predictions into probabilities:

```
σ(z) = 1 / (1 + e^(-z))
```

Where

```
z = w1x1 + w2x2 + ... + wn + b
```

### Training

The model is trained using **gradient descent** to minimize the binary cross-entropy loss.

---

# Model Evaluation

Performance is evaluated using:

* Accuracy
* Confusion Matrix

These metrics help measure how well the model predicts whether an anime will achieve a high score.

---

# Key Insights

Some interesting findings from the analysis:

* **Audience engagement (favorites and members)** is one of the strongest signals for anime success.
* **Popularity and ratings activity** are highly correlated.
* **Episode count alone does not strongly determine ratings.**
* **Genres and source material** contribute meaningful predictive signals.

---

# Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn (for evaluation utilities)

---


# Future Improvements

Possible extensions for this project:

* using weighted cost functiona and weighted gradient
* Train additional models (Random Forest, XGBoost)
* Add studio-level features (studio success rate)
* Build an interactive **Streamlit web app**
* Deploy the model for real-time predictions

---

# Conclusion

This project demonstrates how **machine learning can analyze audience behavior and metadata to predict anime success**.

By combining exploratory data analysis, feature engineering, and a custom logistic regression implementation, the model provides insights into what factors influence highly rated anime.
