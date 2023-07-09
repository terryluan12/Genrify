from datasources import download_datasets
from preprocessing import preprocess
if __name__ == "__main__":
    # Download datasets
    download_datasets()
    
    preprocess("datasources/Data")
    
    
