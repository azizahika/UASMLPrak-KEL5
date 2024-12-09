# 1. Import library tambahan untuk evaluasi
import sys
import seaborn as sns  # Import seaborn untuk heatmap
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 2. Membaca dataset
file_path = 'simplified_coffee.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset berhasil dibaca!")
except FileNotFoundError:
    print(f"File {file_path} tidak ditemukan.")
    exit()

# 3. Menampilkan isi dataset sebelum pengecekan null
print("Dataset sebelum pengecekan null:")
print(df.head())  # Menampilkan 5 baris pertama dataset
print(f"\nJumlah data awal: {df.shape}")

# 4. Mengecek data kosong
print("\nJumlah data kosong per kolom sebelum pemrosesan:")
print(df.isnull().sum())

# 5. Menghapus baris dengan data kosong
df_cleaned = df.dropna()
print("\nJumlah data setelah menghapus baris dengan NaN:")
print(df_cleaned.isnull().sum())

# 6. Mengisi nilai NaN dengan 0 (jika ada yang tersisa)
df_filled = df_cleaned.fillna(0)
print("\nJumlah data kosong setelah pengisian NaN dengan 0:")
print(df_filled.isnull().sum())

# 7. Bentuk data sebelum dan sesudah penghapusan/pengisian NaN
print(f"\nBentuk data sebelum pemrosesan: {df.shape}")
print(f"Bentuk data setelah pemrosesan: {df_filled.shape}")

# 8. Membuat Pivot Table untuk mendapatkan Rating Per Kopi
user_ratings = df_filled.pivot_table(index='name', columns='roaster', values='rating', aggfunc='mean')

# 9. Menghapus kolom yang hanya memiliki NaN (tidak ada rating dari roaster)
user_ratings = user_ratings.dropna(axis=1, how='all')

# 10. Normalisasi data
scaler = StandardScaler()
user_ratings_scaled = pd.DataFrame(scaler.fit_transform(user_ratings.fillna(0)), 
                                   index=user_ratings.index, columns=user_ratings.columns)

# 11. Menghitung Cosine Similarity antar kopi berdasarkan rating pengguna
cosine_sim = cosine_similarity(user_ratings_scaled) 
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_ratings.index, columns=user_ratings.index)

# 12. Fungsi rekomendasi kopi
def recommend_coffee(coffee_name, cosine_sim_df, top_n=5):
    try:
        if coffee_name not in cosine_sim_df.index:
            print(f"Kopi '{coffee_name}' tidak ditemukan dalam dataset.")
            return None
        sim_scores = cosine_sim_df[coffee_name].sort_values(ascending=False)
        similar_coffees = sim_scores.iloc[1:top_n+1]
        return similar_coffees
    except KeyError:
        print(f"Kopi '{coffee_name}' tidak ditemukan dalam dataset.")
        return None

# 13. Input nama kopi dari pengguna
coffee_name = input("Masukkan nama kopi yang ingin Anda rekomendasikan: ").strip()

# 14. Meminta jumlah rekomendasi yang diinginkan
try:
    top_n = int(input("Berapa banyak rekomendasi yang Anda inginkan? (misalnya 5): ").strip())
except ValueError:
    print("Input tidak valid, menggunakan default 5 rekomendasi.")
    top_n = 5

# 15. Mendapatkan rekomendasi berdasarkan input pengguna
recommendations = recommend_coffee(coffee_name, cosine_sim_df, top_n)

# 16. Menampilkan hasil rekomendasi
if recommendations is not None:
    print(f"\nRekomendasi kopi untuk '{coffee_name}':")
    print(recommendations)

# 17. Evaluasi berdasarkan data uji (opsional)
# Anda bisa menghapus atau menyederhanakan evaluasi berikut jika tidak diperlukan
test_data = [
    {"coffee_name": "Ethiopia Shakiso Mormora", "true_recommendations": ["Ethiopia Suke Quto", "Ethiopia Gedeb Halo Beriti"], "top_n": 2},
    {"coffee_name": "Brazil Fazenda", "true_recommendations": ["Kenya AA", "Ethiopia Yirgacheffe"], "top_n": 2},
    # Tambah data uji sesuai kebutuhan
]

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for test in test_data:
    coffee_name = test["coffee_name"]
    true_recommendations = test["true_recommendations"]
    top_n = test["top_n"]
    
    recommendations = recommend_coffee(coffee_name, cosine_sim_df, top_n)
    
    if recommendations is not None:
        recommended_set = set(recommendations.index)
        true_set = set(true_recommendations)
        
        y_true = [1 if coffee in true_set else 0 for coffee in recommended_set]
        y_pred = [1 if coffee in recommended_set else 0 for coffee in true_set]
        
        if len(y_true) != len(y_pred):
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
        
        if len(y_true) > 0 and len(y_pred) > 0:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=1)
            recall = recall_score(y_true, y_pred, zero_division=1)
            f1 = f1_score(y_true, y_pred, zero_division=1)
        else:
            accuracy = precision = recall = f1 = 0.0
        
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

print("\nEvaluasi sistem rekomendasi:")
print(f"Akurasi: {sum(accuracy_list) / len(accuracy_list):.2f}")
print(f"Precision: {sum(precision_list) / len(precision_list):.2f}")
print(f"Recall: {sum(recall_list) / len(recall_list):.2f}")
print(f"F1-Score: {sum(f1_list) / len(f1_list):.2f}")

# 18. Visualisasi Cosine Similarity Menggunakan Heatmap (opsional)
subset_size = 10
subset_sim_df = cosine_sim_df.iloc[:subset_size, :subset_size]

plt.figure(figsize=(10, 8))
sns.heatmap(subset_sim_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Cosine Similarity Heatmap antara Kopi (Subset)')
plt.xlabel('Kopi')
plt.ylabel('Kopi')
plt.show()
