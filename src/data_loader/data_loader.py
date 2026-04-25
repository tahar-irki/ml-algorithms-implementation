import kagglehub
import shutil
import os
import time

def find_data_dir(start_path):
    curr = os.path.abspath(start_path)
    while curr != os.path.dirname(curr):
        potential_data_path = os.path.join(curr, 'data')
        if os.path.isdir(potential_data_path):
            return potential_data_path
        curr = os.path.dirname(curr)
    return None

DATA_DIR = find_data_dir(__file__)

if DATA_DIR is None:
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    print("⚠️ 'data' folder not found, creating one here:", DATA_DIR)

DATASETS = [
    "algozee/teenager-menthal-healy",
    "furkanark/premier-league-2024-2025-data",
    "meharshanali/student-dropout-prediction-dataset",
    "abdallahwagih/spam-emails"
]

os.makedirs(DATA_DIR, exist_ok=True)

for dataset in DATASETS:
    print(f"\n📥 Downloading dataset: {dataset}")
    
    try:
        tmp_path = kagglehub.dataset_download(dataset)
        print("Downloaded to:", tmp_path)
    except Exception as e:
        print(f"❌ Failed to download {dataset}: {e}")
        continue

    found_files = False  # ✅ Reset before the walk

    for root, dirs, files in os.walk(tmp_path):
        for file_name in files:
            found_files = True  # ✅ Mark as found when we actually find a file

            source = os.path.join(root, file_name)
            destination = os.path.join(DATA_DIR, file_name)

            if os.path.exists(destination):
                base, ext = os.path.splitext(file_name)
                destination = os.path.join(
                    DATA_DIR,
                    f"{base}_{dataset.split('/')[-1]}{ext}"
                )

            shutil.move(source, destination)
            print(f"✅ Moved: {file_name}")

    # ✅ Check AFTER the full walk, not inside it
    if not found_files:
        print(f"⚠️ No files found for {dataset}")

    print("⏳ Waiting 10 seconds to avoid rate limits...")
    time.sleep(10)

print(f"\n🎉 Success! All datasets are now in: {DATA_DIR}")