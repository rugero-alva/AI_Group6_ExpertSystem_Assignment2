# app.py (Example using Streamlit for a simple career recommendation)
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Sample career data (replace with your actual data)
career_data = pd.DataFrame({
    'Career': ['Software Engineer', 'Data Scientist', 'Marketing Manager', 'Teacher', 'Nurse'],
    'Programming': [8, 6, 2, 0, 0],
    'Math': [7, 9, 3, 1, 2],
    'Communication': [6, 7, 8, 9, 7],
    'Empathy': [3, 4, 6, 8, 9]
})

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

    skills = data[['Programming', 'Math', 'Communication', 'Empathy']]
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(skills)

    distances, indices = knn.kneighbors([user_skills])
    recommendations = data.iloc[indices[0]]['Career'].tolist()
    return recommendations

st.title("Student Career Guidance")

st.write("Please rate your skills (1-10):")

programming_skill = st.slider("Programming", 1, 10, 5)
math_skill = st.slider("Math", 1, 10, 5)
communication_skill = st.slider("Communication", 1, 10, 5)
empathy_skill = st.slider("Empathy", 1, 10, 5)

if st.button("Get Recommendations"):
    user_skills = [programming_skill, math_skill, communication_skill, empathy_skill]
    recommendations = get_recommendations(user_skills, career_data)

    st.write("Recommended Careers:")
    for career in recommendations:
        st.write("- " + career)

# requirements.txt
streamlit
pandas
scikit-learn