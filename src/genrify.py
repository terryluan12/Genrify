from datasources import download_datasets
from preprocessing import preprocess
if __name__ == "__main__":
    # Download datasets
    if not os.path.isdir("datasources/Data"):
        download_datasets()
    
    preprocess("datasources/Data")
    
    
