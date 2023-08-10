import os
import requests
from zipfile import ZipFile
from pydub import AudioSegment

def download_datasets(root_dir="."):
    fname = os.path.join(root_dir, "datasources", "music.zip")
    url = "https://osf.io/drjhb/download"
    print("Downloading Datasources...")

    try:
        r = requests.get(url)
    except requests.ConnectionError:
        print("!!! Failed to download data !!!")
    else:
        if r.status_code != requests.codes.ok:
            print("!!! Failed to download data !!!")
        else:
            with open(fname, "wb") as fid:
                fid.write(r.content)
                
    
    with ZipFile(fname, 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        zipObj.extractall(os.path.join(root_dir, "datasources"))
    os.remove(fname)
    print(f"Downloaded Datasources")

def convert_to_wav(input_file, output_file):
    sound = AudioSegment.from_mp3(input_file)
    sound.export(output_file, format="wav")
    print(f"Conversion successful: {input_file} -> {output_file}")

def convert_to_mp3(input_file, output_file):
    sound = AudioSegment.from_wav(input_file)
    sound.export(output_file, format="mp3")
    print(f"Conversion successful: {input_file} -> {output_file}")

def convert_files(data_dir, output_dir, wavToMp3 = False):
    os.makedirs(output_dir, exist_ok=True)
    for genre in os.listdir(data_dir):
        genre_dir = os.path.join(data_dir, genre)
        if os.path.isdir(genre_dir):
            wav_dir = os.path.join(output_dir, f"{genre}")
            os.makedirs(wav_dir, exist_ok=True)
            for filename in os.listdir(genre_dir):
                if wavToMp3 == False:
                    if filename.endswith(".mp3"):
                        input_file = os.path.join(genre_dir, filename)
                        output_file = os.path.join(wav_dir, filename.replace(".mp3", ".wav"))
                        convert_to_wav(input_file, output_file)
                else:
                    if filename.endswith(".wav"):
                        input_file = os.path.join(genre_dir, filename)
                        output_file = os.path.join(wav_dir, filename.replace(".wav", ".mp3"))
                        convert_to_wav(input_file, output_file)
