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

    full_dataset = DatasetFolder(datasources_dir, librosa.load, extensions=[".wav"])
    training_indeces = [num for subrange in [range(x, x+700) for x in range(0, 1000, 100)] for num in subrange]
    valid_indeces = [num for subrange in [range(x+700, x+700+150) for x in range(0, 1000, 100)] for num in subrange]
    test_indeces = [num for subrange in [range(x+700+150, x+1000) for x in range(0, 1000, 100)] for num in subrange]
    
    full_training = torch.utils.data.Subset(full_dataset, training_indeces)
    full_valid = torch.utils.data.Subset(full_dataset, valid_indeces)
    full_test = torch.utils.data.Subset(full_dataset, test_indeces)

    subsets_ratio = 1./num_subsets
    training_size = int(subsets_ratio * len(full_training))
    valid_size = int(subsets_ratio * len(full_valid))
    test_size = int(subsets_ratio * len(full_test))

    training_subsets = [torch.utils.data.random_split(full_training, [training_size]*num_subsets)]
    valid_subsets = [torch.utils.data.random_split(full_valid, [valid_size]*num_subsets)]
    test_subsets = [torch.utils.data.random_split(full_test, [test_size]*num_subsets)]

    
    return [training_subsets, valid_subsets, test_subsets]

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