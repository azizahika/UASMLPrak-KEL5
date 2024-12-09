from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

# Membaca dataset
file_path = 'simplified_coffee.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset berhasil dibaca!")
except FileNotFoundError:
    print(f"File {file_path} tidak ditemukan.")
    exit()

# Preprocess dataset
df_cleaned = df.dropna()
df_filled = df_cleaned.fillna(0)
user_ratings = df_filled.pivot_table(index='name', columns='roaster', values='rating', aggfunc='mean')
user_ratings = user_ratings.dropna(axis=1, how='all')
scaler = StandardScaler()
user_ratings_scaled = pd.DataFrame(scaler.fit_transform(user_ratings.fillna(0)),
                                   index=user_ratings.index, columns=user_ratings.columns)
cosine_sim = cosine_similarity(user_ratings_scaled)
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_ratings.index, columns=user_ratings.index)

# Fungsi rekomendasi kopi
def recommend_coffee(coffee_name, cosine_sim_df, top_n=5):
    try:
        sim_scores = cosine_sim_df[coffee_name].sort_values(ascending=False)
        similar_coffees = sim_scores.iloc[1:top_n+1]
        return similar_coffees
    except KeyError:
        return None

# Fungsi untuk menghitung evaluasi akurasi
def evaluate_recommendations(test_data, cosine_sim_df):
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

    return {
        "accuracy": sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0,
        "precision": sum(precision_list) / len(precision_list) if precision_list else 0,
        "recall": sum(recall_list) / len(recall_list) if recall_list else 0,
        "f1": sum(f1_list) / len(f1_list) if f1_list else 0
    }

# Contoh data uji untuk evaluasi
test_data = [
    {"coffee_name": "Ethiopia Shakiso Mormora", "true_recommendations": ["Ethiopia Suke Quto", "Ethiopia Gedeb Halo Beriti"], "top_n": 2},
    {"coffee_name": "Brazil Fazenda", "true_recommendations": ["Kenya AA", "Ethiopia Yirgacheffe"], "top_n": 2},
    # Tambah data uji sesuai kebutuhan
]

# Rute untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Rute untuk halaman kopi
@app.route('/coffees')
def coffees():
    return render_template('coffees.html')

# Rute untuk halaman review
@app.route('/review')  
def review():
    return render_template('review.html') 

# Rute untuk halaman rekomendasi
@app.route('/rekomendasi')
def contact():
    return render_template('rekomendasi.html')

@app.route('/rekomendasi', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        coffee_name = request.form.get('coffee_name')
        top_n = int(request.form.get('top_n', 5))
        
        recommendations = recommend_coffee(coffee_name, cosine_sim_df, top_n)
        
        if recommendations is not None:
            rec_list = recommendations.index.tolist()

            # Evaluasi sistem rekomendasi
            eval_results = evaluate_recommendations(test_data, cosine_sim_df)

            # Membuat Heatmap Cosine Similarity
            subset_size = 10
            subset_sim_df = cosine_sim_df.iloc[:subset_size, :subset_size]
            plt.figure(figsize=(10, 8))
            sns.heatmap(subset_sim_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Cosine Similarity Heatmap antara Kopi (Subset)')
            plt.xlabel('Kopi')
            plt.ylabel('Kopi')

            # Menyimpan gambar heatmap ke dalam format base64
            img_stream = io.BytesIO()
            plt.savefig(img_stream, format='png')
            img_stream.seek(0)
            heatmap_base64 = base64.b64encode(img_stream.getvalue()).decode()

            return render_template('rekomendasi.html', coffee_name=coffee_name, recommendations=rec_list, eval_results=eval_results, heatmap_base64=heatmap_base64)
        else:
            error_message = f"Kopi '{coffee_name}' tidak ditemukan dalam dataset."
            return render_template('rekomendasi.html', coffee_name=None, error_message=error_message)
    
    return render_template('rekomendasi.html')

if __name__ == '__main__':
    app.run(debug=True)
