import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

if 'title' not in movies.columns or 'genres' not in movies.columns:
    raise ValueError("Dataset must contain 'title' and 'genres' columns.")
if 'movieId' not in ratings.columns or 'rating' not in ratings.columns:
    raise ValueError("Ratings dataset must contain 'movieId' and 'rating' columns.")

movies["combined_features"] = movies["title"] + " " + movies["genres"]

cv = CountVectorizer()
vectorized_matrix = cv.fit_transform(movies["combined_features"])

similarity_scores = cosine_similarity(vectorized_matrix)

average_ratings = ratings.groupby("movieId")["rating"].mean()

movies = movies.merge(average_ratings, left_on="movieId", right_on="movieId", how="left")
movies["rating"] = movies["rating"].fillna(0)

def recommend(movie_title, num_recommendations=5):
    try:
        movie_index = movies[movies["title"] == movie_title].index[0]
        similar_movies = list(enumerate(similarity_scores[movie_index]))
        sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        recommendations = []
        for i in sorted_movies[1:num_recommendations + 1]:
            title = movies.loc[i[0], "title"]
            rating = movies.loc[i[0], "rating"]
            genres = movies.loc[i[0], "genres"]
            recommendations.append((title, rating, genres))
        return recommendations
    except IndexError:
        return ["Movie not found in the dataset. Please check the title and try again."]

if __name__ == "__main__":
    movie_title = input("Enter a movie title to get recommendations: ")
    num_recommendations = int(input("Enter the number of recommendations you want: "))
    recommendations = recommend(movie_title, num_recommendations)

    print("\nRecommended movies:")
    for rec in recommendations:
        if isinstance(rec, str):
            print(rec)
        else:
            movie, rating, genres = rec
            print(f"- {movie} (Rating: {rating:.1f}, Genres: {genres})")
