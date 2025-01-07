import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

st.title('🎥 movie recommendation')

st.write('welcome lol')
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

movies["combined_features"] = movies["title"] + " " + movies["genres"]

cv = CountVectorizer()
vectorized_matrix = cv.fit_transform(movies["combined_features"])
similarity_scores = cosine_similarity(vectorized_matrix)

ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
model = TruncatedSVD(n_components=50)
latent_matrix = model.fit_transform(ratings_matrix)

latent_matrix_df = pd.DataFrame(latent_matrix, index=ratings_matrix.index)

similarity_matrix = cosine_similarity(latent_matrix_df)

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
    if user_id not in latent_matrix_df.index:
        return ["User not found in the dataset. Please check the user ID and try again."]

    user_similarity_scores = similarity_matrix[user_id - 1]
    similar_users = list(enumerate(user_similarity_scores))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)

    recommended_movies = {}
    for similar_user, _ in similar_users[:10]:
        user_ratings = ratings_matrix.iloc[similar_user]
        for movie_id in user_ratings[user_ratings > 4].index:
            if movie_id not in recommended_movies:
                recommended_movies[movie_id] = user_ratings[movie_id]

    recommended_movies = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)

    recommendations = []
    for movie_id, _ in recommended_movies[:num_recommendations]:
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
