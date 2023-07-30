import os
from pydub import AudioSegment
from torchvision.datasets import DatasetFolder
import torch
import librosa

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


def split_into_exclusive_datasets(datasources_dir="datasources/processed_data", num_subsets=4):
    print(f"Splitting processed Data into {num_subsets} exclusive datasets.")
    torch.manual_seed(42)
    subsets = []
    subset_split = [0.7, 0.15, 0.15]

    divide = 1./num_subsets
    equal_lengths = [divide for _ in range(num_subsets)]

    full_dataset = DatasetFolder(datasources_dir, librosa.load, extensions=[".wav"])
    full_subsets = torch.utils.data.random_split(full_dataset, equal_lengths)
    for full_subset in full_subsets:
        subset = torch.utils.data.random_split(full_subset, subset_split)
        subsets.append(subset)

    return subsets

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
                    origin_file = os.path.join(origin_dir, file)
                    try:
                        sound = AudioSegment.from_wav(origin_file)
                    except:
                        print(f'Skipping {origin_file}')
                        continue
                    num_segments = int(45 / 3) #Skip first 10 seconds and save 45 seconds worth
                    for i in range(num_segments):
                        extract = sound[(i * 3000)+10000 : ((i + 1) * 3000)+10000]
                        extract.export(os.path.join(destination_dir, file + "_trimmed" + str(i) + ".wav"), format="wav")