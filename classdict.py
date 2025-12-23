import pickle

paths = [
    r"D:\Skin Disease Detection\model\cnn_skin_disease.pkl",
    r"D:\Skin Disease Detection\model\convnext_base_skin_disease.pkl",
    r"D:\Skin Disease Detection\model\resnet101_skin_disease.pkl",
]

for path in paths:
    print(f"\n=== Testing {path} ===")
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print("Berhasil dibaca dengan pickle.load()")
        print("Tipe data:", type(data))
        if isinstance(data, dict):
            print("Keys:", data.keys())
        else:
            print("Ini model lengkap (bukan dict)")
            print("Apakah ada attribute .fc atau .head?", hasattr(data, 'fc') or hasattr(data, 'head'))
    except Exception as e:
        print("GAGAL:", str(e))