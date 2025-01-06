# 🎥 Movie Recommendation System

This project implements a **Movie Recommendation System** using **Python** and **machine learning techniques**. It provides recommendations based on the similarity of movie features such as titles and genres.

## Features
- **Content-Based Filtering**: Recommends movies by analyzing the features of the movies.
- **Customizable**: Extendable to include additional features like cast, director, or user ratings.
- **Simple and Intuitive**: Built using straightforward methods for quick prototyping.

## Dataset
The project uses the [MovieLens Dataset](https://grouplens.org/datasets/movielens/) provided by GroupLens Research. 
- You can download the **ml-latest-small** dataset for a smaller version.

## Requirements
Make sure you have the following installed:
- Python 3.7+
- Libraries: 
  - pandas
  - scikit-learn

Install the required libraries using pip:
```bash
pip install pandas scikit-learn
```
or
```bash
pip install -r /path/to/requirements.txt
```
## How It Works
1. **Data Preprocessing**: 
   - Combine relevant features (e.g., title, genres) into a single column.
   - Clean and preprocess the data.

2. **Vectorization**:
   - Use `CountVectorizer` to convert text data into numerical form.

3. **Compute Similarity**:
   - Use `cosine_similarity` to calculate the similarity between movies.

4. **Recommendation**:
   - Recommend top `n` similar movies based on a given movie title.

## Usage
### Clone the Repository
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

### Run the Project
1. Download the dataset and place it in the project directory.
2. Open the `movie_recommendation.py` file and update the dataset path.
3. Run the script:
   ```bash
   python movie_recommendation.py
   ```

### Example Code
```python
print(recommend("The Godfather", 5))
```
This will return the top 5 movies similar to *The Godfather*.

## Project Structure
```
movie-recommendation-system/
├── movies.csv             # Dataset
├── movie_recommendation.py # Main program code
├── README.md              # Documentation
```

## Made by Btin7
