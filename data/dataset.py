import kagglehub
import os
import shutil

local_dir = r"D:\Skin Disease Detection\Data"
os.makedirs(local_dir, exist_ok=True)
cache_path = kagglehub.dataset_download("pacificrm/skindiseasedataset")

for item in os.listdir(cache_path):
    src = os.path.join(cache_path, item)
    dst = os.path.join(local_dir, item)

    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Dataset berhasil disimpan di:", os.path.abspath(local_dir))