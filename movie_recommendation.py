import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title('🎥 Movie Recommendation')

st.write('Welcome!')

def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

movies["combined_features"] = movies["title"] + " " + movies["genres"]

cv = CountVectorizer()
vectorized_matrix = cv.fit_transform(movies["combined_features"])
similarity_scores = cosine_similarity(vectorized_matrix)

def recommend_content(movie_title, genre_filter=None, num_recommendations=5):
    try:
        movie_index = movies[movies["title"].str.lower() == movie_title.lower()].index[0]
        similar_movies = list(enumerate(similarity_scores[movie_index]))
        sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        recommendations = []

        for i in sorted_movies[1:]:
            title = movies.loc[i[0], "title"]
            genres = movies.loc[i[0], "genres"]
            if genre_filter and genre_filter.lower() not in genres.lower():
                continue
            avg_rating = ratings[ratings['movieId'] == movies.loc[i[0], 'movieId']]['rating'].mean()
            recommendations.append((title, genres, avg_rating))
            if len(recommendations) >= num_recommendations:
                break
        return recommendations
    except IndexError:
        return ["Movie not found in the dataset. Please check the title and try again."]

st.sidebar.header("Choose Recommendation Type")
recommendation_type = st.sidebar.selectbox("Recommendation Type", ["Content-Based", "Genre-Based"])

if recommendation_type == "Content-Based":
    st.header("Content-Based Recommendations")
    movie_title = st.text_input("Enter a movie title")
    genre_filter = st.text_input("Filter by genre (optional)")
    num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)

    if st.button("Recommend"):
        recommendations = recommend_content(movie_title, genre_filter, num_recommendations)
        st.write("Recommendations:")
        for rec in recommendations:
            if isinstance(rec, str):
                st.write(rec)
            else:
                movie, genres, avg_rating = rec
                st.write(f"- {movie} (Genres: {genres}, Rating: {avg_rating:.2f})")

elif recommendation_type == "Genre-Based":
    st.header("Genre-Based Recommendations")
    genre = st.text_input("Enter a genre")
    num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)

    if st.button("Recommend"):
        filtered_movies = movies[movies['genres'].str.contains(genre, case=False, na=False)]
        if len(filtered_movies) > 0:
            recommendations = filtered_movies.sample(num_recommendations) if num_recommendations <= len(filtered_movies) else filtered_movies
            st.write("Recommendations:")
            for index, row in recommendations.iterrows():
                avg_rating = ratings[ratings['movieId'] == row['movieId']]['rating'].mean()
                st.write(f"- {row['title']} (Genres: {row['genres']}, Rating: {avg_rating:.2f})")
        else:
            st.write("No movies found for the selected genre.")
