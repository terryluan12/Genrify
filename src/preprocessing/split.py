import os
from pydub import AudioSegment
from torchvision.datasets import DatasetFolder
import torch
import librosa
import random

def split_into_3_seconds(datasources_dir="datasources"):
    print(f'Splitting full music Data into 3 second chunks')
    error_converted_files = []
    data_dir = os.path.join(datasources_dir, "Data", "genres_original")
    processed_dir = os.path.join(datasources_dir,"processed_data")
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)

    for _, dirs, _ in os.walk(data_dir):
        for dir in dirs:
            print(f"Traversing {dir}")
            origin_dir = os.path.join(data_dir, dir)
            destination_dir = os.path.join(processed_dir, dir)
            if not os.path.isdir(destination_dir):
                os.mkdir(destination_dir)
            for _, _, files in os.walk(origin_dir):
                for file in files:
                    origin_file = os.path.join(origin_dir, file)
                    try:
                        sound = AudioSegment.from_wav(origin_file)
                    except:
                        print(f'Skipping {origin_file}')
                        continue
                    num_segments = int(sound.duration_seconds / 3)
                    for i in range(num_segments):
                        extract = sound[i * 3000 : (i + 1) * 3000]
                        extract.export(os.path.join(destination_dir, file + "_trimmed" + str(i) + ".wav"), format="wav")

def shuffle_subsets(subsets):
    num_subsets = len(subsets)
    indeces = list(range(num_subsets))
    for i in range(len(subsets[0])):
        perm_indeces = random.sample(indeces, len(indeces))
        swap_values = [subsets[x][i] for x in range(num_subsets)]
        for j in range(num_subsets):
            subsets[j][i] = swap_values[perm_indeces[j]]

def split_into_exclusive_datasets(datasources_dir="datasources/processed_data", num_subsets=4):
    torch.manual_seed(42)

    print(f"Splitting processed Data into {num_subsets} exclusive datasets.")

    full_dataset = DatasetFolder(datasources_dir, librosa.load, extensions=[".mp3"])
    
    num_genres = 10
    genre_length = 1000
    samples_per_genre = int(genre_length/num_subsets)
    subset_indeces = []
    for i in range(num_subsets):
        indeces = [i*samples_per_genre + j*genre_length + k for j in range(num_genres) for k in range(samples_per_genre)]
        subset_indeces.append(indeces)
    
    shuffle_subsets(subset_indeces)

    datasets = []
    for i in range(num_subsets):
        datasets.append(torch.utils.data.Subset(full_dataset, subset_indeces[i]))
    
    return datasets

def split_test_data_into_3_seconds(datasources_dir="datasources"):
    print(f'Splitting test music Data into 3 second chunks')
    error_converted_files = []
    data_dir = os.path.join(datasources_dir, "test_data_wav")
    processed_dir = os.path.join(datasources_dir,"processed_test_data")
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)

    for _, dirs, _ in os.walk(data_dir):
        for dir in dirs:
            print(f"Traversing {dir}")
            origin_dir = os.path.join(data_dir, dir)
            destination_dir = os.path.join(processed_dir, dir)
            if not os.path.isdir(destination_dir):
                os.mkdir(destination_dir)
            for _, _, files in os.walk(origin_dir):
                for file in files:
                    _, extension = os.path.splitext(file)
                    origin_file = os.path.join(origin_dir, file)
                    if extension == ".mp3":
                        try:
                            sound = AudioSegment.from_mp3(origin_file)
                        except:
                            print(f'Skipping {origin_file}')
                            continue
                    elif extension == ".wav":
                        try:
                            sound = AudioSegment.from_wav(origin_file)
                        except:
                            print(f'Skipping {origin_file}')
                            continue
                    else:
                        print(f'Skipping {origin_file}, not a wav or mp3 file')
                        continue
                    
                    # Skip first 10 seconds and save 45 seconds worth
                    num_segments = int(45 / 3)
                    for i in range(num_segments):
                        extract = sound[(i * 3000)+10000 : ((i + 1) * 3000)+10000]
                        extract.export(os.path.join(destination_dir, file + "_trimmed" + str(i) + ".wav"), format="wav")