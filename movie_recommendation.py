import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split

def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

movies["combined_features"] = movies["title"] + " " + movies["genres"]

cv = CountVectorizer()
vectorized_matrix = cv.fit_transform(movies["combined_features"])
similarity_scores = cosine_similarity(vectorized_matrix)

reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
trainset, _ = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)

def recommend_content(movie_title, num_recommendations=5):
    try:
        movie_index = movies[movies["title"].str.lower() == movie_title.lower()].index[0]
        similar_movies = list(enumerate(similarity_scores[movie_index]))
        sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        recommendations = []
        for i in sorted_movies[1:num_recommendations + 1]:
            title = movies.loc[i[0], "title"]
            genres = movies.loc[i[0], "genres"]
            recommendations.append((title, genres))
        return recommendations
    except IndexError:
        return ["Movie not found in the dataset. Please check the title and try again."]

def recommend_collaborative(user_id, num_recommendations=5):
    movie_ids = movies['movieId'].unique()
    predictions = []
    for movie_id in movie_ids:
        pred = model.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = predictions[:num_recommendations]

    recommendations = []
    for movie_id, _ in top_recommendations:
        movie = movies[movies['movieId'] == movie_id]
        recommendations.append((movie['title'].values[0], movie['genres'].values[0]))

    return recommendations

st.title("Movie Recommendation System")
st.sidebar.header("Choose Recommendation Type")
recommendation_type = st.sidebar.selectbox("Recommendation Type", ["Content-Based", "Collaborative Filtering", "Hybrid"])

if recommendation_type == "Content-Based":
    st.header("Content-Based Recommendations")
    movie_title = st.text_input("Enter a movie title")
    num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)

    if st.button("Recommend"):
        recommendations = recommend_content(movie_title, num_recommendations)
        st.write("Recommendations:")
        for rec in recommendations:
            if isinstance(rec, str):
                st.write(rec)
            else:
                movie, genres = rec
                st.write(f"- {movie} (Genres: {genres})")

elif recommendation_type == "Collaborative Filtering":
    st.header("Collaborative Filtering Recommendations")
    user_id = st.number_input("Enter User ID", min_value=1, step=1)
    num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)

    if st.button("Recommend"):
        recommendations = recommend_collaborative(user_id, num_recommendations)
        st.write("Recommendations:")
        for movie, genres in recommendations:
            st.write(f"- {movie} (Genres: {genres})")

else:
    st.header("Hybrid Recommendations")
    movie_title = st.text_input("Enter a movie title")
    user_id = st.number_input("Enter User ID", min_value=1, step=1)
    num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)

    if st.button("Recommend"):
        content_recommendations = recommend_content(movie_title, num_recommendations // 2)
        collaborative_recommendations = recommend_collaborative(user_id, num_recommendations // 2)

        st.write("Content-Based Recommendations:")
        for rec in content_recommendations:
            if isinstance(rec, str):
                st.write(rec)
            else:
                movie, genres = rec
                st.write(f"- {movie} (Genres: {genres})")

        st.write("Collaborative Filtering Recommendations:")
        for movie, genres in collaborative_recommendations:
            st.write(f"- {movie} (Genres: {genres})")
