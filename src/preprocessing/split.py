import os
from pydub import AudioSegment

def split_into_3_seconds(root_dir="datasources"):
    error_converted_files = []
    data_dir = os.path.join(root_dir, "Data", "genres_original")
    processed_dir = os.path.join(root_dir,"processed_data")
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)

    for _, dirs, _ in os.walk(data_dir):
        for dir in dirs:
            print(f"Traversing {dir}")
            origin_dir = os.path.join(data_dir)
            destination_dir = os.path.join(processed_dir, dir)
            if not os.path.isdir(destination_dir):
                os.mkdir(destination_dir)
            for _, _, files in os.walk(origin_dir):
                print(f"files are {files}")
                for file in files:
                    print(f"\tTraversing {file}")
                    origin_file = os.path.join(origin_dir, dir, file)
                    print(f'Attempting to process {origin_file}')
                    try:
                        sound = AudioSegment.from_wav(origin_file)
                    except:
                        error_converted_files.append(origin_file)
                        print(f'Skipping {origin_file}')
                        continue
                    num_segments = int(sound.duration_seconds / 3)
                    print(f"Duration in seconds is {num_segments}")
                    for i in range(num_segments):
                        print(f"Importing file {i}")
                        extract = sound[(i-1) * 3000 : i * 3000]
                        extract.export(os.path.join(destination_dir, file + "_trimmed" + str(i) + ".wav"), format="wav")
    