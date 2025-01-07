import streamlit as st
import pandas as pd

st.title('🎥 Movie Recommendation')

st.write('Welcome!')

def load_data():
    movies = pd.read_csv("movies.csv")
    return movies

movies = load_data()

st.sidebar.header("Filter by Genre")
genre = st.sidebar.selectbox("Select Genre", movies['genres'].unique())

def recommend_by_genre(selected_genre, num_recommendations=5):
    filtered_movies = movies[movies['genres'].str.contains(selected_genre, case=False, na=False)]
    recommendations = filtered_movies.sample(num_recommendations) if not filtered_movies.empty else []
    return recommendations

st.header("Genre-Based Recommendations")
num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("Recommend"):
    recommendations = recommend_by_genre(genre, num_recommendations)
    st.write("Recommendations:")
    if recommendations:
        for index, row in recommendations.iterrows():
            st.write(f"- {row['title']} (Genres: {row['genres']})")
    else:
        st.write("No movies found for the selected genre.")
