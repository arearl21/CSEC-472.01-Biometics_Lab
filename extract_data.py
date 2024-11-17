import os
import zipfile
import libtorrent as lt
import time

# Path to save the downloaded file
torrent_file = "/home/kali/Downloads/fingerprints.torrent"
output_dir = "/home/kali/fingerprints"

# Download using libtorrent
def download_torrent(torrent_file, output_dir):
    ses = lt.session()
    ses.listen_on(6881, 6891)
    info = lt.torrent_info(torrent_file)
    handle = ses.add_torrent({'ti': info, 'save_path': output_dir})
    print("Starting torrent download...")
    
    while not handle.is_seed():
        s = handle.status()
        print(f"Download Progress: {s.progress * 100:.2f}%")
        time.sleep(1)
    print("Download complete!")

# Extract dataset
def extract_dataset(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")

# Example usage
download_torrent("dataset.torrent", "./dataset_folder")
extract_dataset("./dataset_folder/dataset.zip", "./dataset")
