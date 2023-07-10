import os
from pydub import AudioSegment

def split_into_3_seconds(root_dir):
    error_converted_files = []
    data_dir = root_dir + "/Data/genres_original"
    processed_dir = root_dir + "/processed_data"
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)

    for _, dirs, _ in os.walk(data_dir):
        for dir in dirs:
            converting_dir = os.path.join(processed_dir, dir)
            if not os.path.isdir(converting_dir):
                os.mkdir(converting_dir)
            for _, _, files in os.walk(converting_dir):
                for file in files:
                    converting_file = os.path.join(data_dir, dir, file)
                    print(f'Attempting to process {converting_file}')
                    try:
                        sound = AudioSegment.from_wav(converting_file)
                    except:
                        error_converted_files.append(converting_file)
                        print(f'Skipping {converting_file}')
                        continue
                    num_segments = int(sound.duration_seconds / 3)
                    print(f"Duration in seconds is {num_segments}")
                    for i in range(num_segments):
                        print(f"Importing file {i}")
                        extract = sound[(i-1) * 3000 : i * 3000]
                        extract.export(os.path.join(converting_dir, file + "_trimmed" + str(i) + ".wav"), format="wav")
    print(f'There was an error converting these files {error_converted_files}')