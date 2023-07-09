import opendatasets as od
import shutil

# download gtzan
od.download_kaggle_dataset("https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification", "./")
shutil.rename('gtzan-dataset-music-genre-classification', 'GTZAN')