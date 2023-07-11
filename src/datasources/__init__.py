import os
import requests
from zipfile import ZipFile

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
        zipObj.extractall("datasources")
    os.remove(fname)
    print(f"Downloaded Datasources")
        