import os
from pydub import AudioSegment

def split_into_3_seconds(root_dir):
    error_converted_files = []
    
    if not os.path.isdir("processed_data"):
        os.mkdir("processed_data")

    for _, dirs, _ in os.walk(root_dir):
        for dir in dirs:
            if not os.path.isdir("processed_data/" + dir):
                os.mkdir("processed_data/" + dir)
            for _, _, files in os.walk(root_dir + "/" + dir):
                for file in files:
                    converting_file = root_dir + "/" + dir + "/" + file
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
                        extract.export("processed_data/" + dir + "/" + file + "_trimmed" + str(i) + ".wav", format="wav")
    print(f'There was an error converting these files {error_converted_files}')