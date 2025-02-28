import torch
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    def __init__(self, users, movies, genres, ratings):
        self.users = users
        self.movies = movies
        self.genres = genres
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        users = self.users[idx]
        movies = self.movies[idx]
        genres = self.genres[idx]
        ratings = self.ratings[idx]

        return {
            "users": torch.tensor(users, dtype=torch.long),
            "movies": torch.tensor(movies, dtype=torch.long),
            "genres": torch.tensor(genres, dtype=torch.long),
            "ratings": torch.tensor(ratings, dtype=torch.float),
        }
