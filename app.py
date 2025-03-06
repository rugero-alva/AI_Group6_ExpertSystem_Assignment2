
# The app that will guide students in choosing a career path based on their skills. The app uses a simple k-nearest neighbors (KNN) model to recommend careers based on the user's skill ratings. The user can rate their skills in programming, math, communication, Big Data, Finance, Biology, Chemistry, Physics, Geography, History, English, Ethics, and Empathy. The app then uses the KNN model to find the most similar careers based on the user's skill ratings and recommends them to the user.

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Full career data according to skills
career_data = pd.DataFrame({
    'career': [
        'Software Engineer', 'Data Scientist', 'Marketing Manager', 'Teacher', 'Nurse',
        'Financial Analyst', 'Doctor', 'Civil Engineer', 'Psychologist', 'Lawyer'
    ],
    'Programming': [9, 8, 3, 1, 1, 2, 1, 6, 2, 1],
    'Math': [8, 9, 4, 2, 3, 9, 5, 7, 2, 4],
    'Communication': [6, 7, 9, 9, 8, 5, 7, 6, 8, 9],
    'Empathy': [4, 5, 7, 9, 10, 6, 9, 5, 9, 6],
    'Big Data': [7, 9, 4, 1, 1, 6, 2, 4, 3, 5],
    'Finance': [5, 6, 7, 2, 1, 10, 2, 3, 2, 7],
    'Biology': [2, 3, 2, 1, 9, 3, 10, 2, 7, 2],
    'Chemistry': [3, 4, 2, 1, 8, 4, 9, 2, 5, 1],
    'Physics': [4, 5, 2, 1, 2, 6, 2, 9, 2, 1],
    'Geography': [2, 3, 2, 8, 2, 3, 2, 5, 2, 2],
    'History': [2, 3, 7, 9, 3, 3, 2, 3, 8, 9],
    'English': [3, 4, 8, 9, 7, 5, 6, 4, 7, 9],
    'Ethics': [3, 5, 6, 8, 9, 4, 8, 3, 9, 7],
})

# Function to get recommendations based on user skills
def get_recommendations(user_skills, data, k=3):
    """
    Recommends careers based on user skills.

    Args:
        user_skills (list): List of user's skill ratings.
        data (DataFrame): Career data.
        k (int): Number of recommendations.

    Returns:
        list: List of recommended careers.
    """
    skills = data.drop(columns=['career'])  # Remove career names
    scaler = StandardScaler()
    scaled_skills = scaler.fit_transform(skills)  # Normalize skill values
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(scaled_skills)

    user_skills_scaled = scaler.transform([user_skills])
    distances, indices = knn.kneighbors(user_skills_scaled)
    
    recommendations = data.iloc[indices[0]]['career'].tolist()
    return recommendations

# Streamlit app layout
st.title("🎓 Student Career Guidance App")
st.write("Rate your skills on a scale of 1 to 10 to receive career recommendations.")

# Creating skill sliders
skills_labels = [
    "Programming", "Math", "Communication", "Empathy", "Big Data", "Finance",
    "Biology", "Chemistry", "Physics", "Geography", "History", "English", "Ethics"
]

user_skills = []
for skill in skills_labels:
    user_skills.append(st.slider(skill, 1, 10, 5))

# Recommendation button
if st.button("Get Career Recommendations"):
    recommendations = get_recommendations(user_skills, career_data)

    st.success("✅ Based on your skills, these careers are best suited for you:")
    for i, career in enumerate(recommendations, 1):
        st.write(f"{i}. {career}")

# Reset button
if st.button("Reset"):
    st.experimental_rerun()

