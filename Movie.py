# sentiment_analysis.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv('movie_reviews.csv')
X = data['review']
y = data['sentiment']

# Preprocess and vectorize text data
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))


# user_clustering.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('user_ratings.csv')

# Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(data)

# Apply clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

# Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.title('User Clusters')
plt.show()

# reinforcement_learning.py

import numpy as np
import pandas as pd
import random

# Load dataset
ratings = pd.read_csv('user_ratings.csv')

# Define Q-learning parameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Initialize Q-table
Q = np.zeros((n_states, n_actions))

# Define training process
for episode in range(n_episodes):
    state = random.choice(states)
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, done = take_action(state, action)
        
        old_value = Q[state, action]
        next_max = np.max(Q[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        Q[state, action] = new_value
        
        state = next_state
