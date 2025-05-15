import pandas as pd
import numpy as np

np.random.seed(42)
n = 125  # mỗi nhóm 125 khách

def generate_group(prefix, tuoi_range, tn_range, chi_range, mua_range, diem_range, vip_range, luot_range, diemhh_range):
    return {
        "MaKH": [f"{prefix}_{i}" for i in range(1, n + 1)],
        "Tuoi": np.random.randint(*tuoi_range, size=n),
        "ThuNhap": np.random.randint(*tn_range, size=n),
        "ChiTieuHangThang": np.random.randint(*chi_range, size=n),
        "SoLanMua": np.random.randint(*mua_range, size=n),
        "DiemTinDung": np.random.randint(*diem_range, size=n),
        "CapDoVIP": np.random.randint(*vip_range, size=n),
        "LuotTruyCapWeb": np.random.randint(*luot_range, size=n),
        "DiemHoiHoi": np.random.randint(*diemhh_range, size=n)
    }

# 4 nhóm khách hàng với đặc điểm khác nhau
g1 = generate_group("G1", (25, 40), (80000, 100000), (40000, 80000), (10, 30), (700, 850), (3, 5), (10, 100), (500, 1000))
g2 = generate_group("G2", (30, 50), (30000, 50000), (5000, 15000), (5, 15), (400, 650), (1, 3), (50, 200), (100, 400))
g3 = generate_group("G3", (18, 35), (8000, 20000), (3000, 7000), (30, 60), (300, 500), (0, 2), (100, 300), (50, 150))
g4 = generate_group("G4", (35, 60), (60000, 80000), (20000, 40000), (15, 35), (650, 800), (2, 4), (20, 150), (300, 800))

df = pd.concat([pd.DataFrame(g) for g in [g1, g2, g3, g4]], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Lưu file
file_path = "D:/LUUDULIEU/CODE/CURE/dulieu/khachhang_500_8truong.csv"
df.to_csv(file_path, index=False)

file_path
