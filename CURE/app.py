from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import json

app = Flask(__name__)

try:
    from cure import CURE
except ImportError:
    from sklearn.cluster import KMeans
    CURE = None

def preprocess_data(df, fields):
    # Tính toán Q1, Q3 và IQR để loại bỏ outliers
    Q1 = df[fields].quantile(0.25)
    Q3 = df[fields].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[~((df[fields] < (Q1 - 1.5 * IQR)) | (df[fields] > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # Chuẩn hóa dữ liệu bằng RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_clean[fields])
    
    return X_scaled, df_clean

def optimize_clustering(X_scaled, num_clusters, n_represent):
    # Khởi tạo các biến để lưu kết quả tốt nhất
    best_score = -1
    best_compression = 0.3
    best_labels = None
    
    # Thử nghiệm các giá trị compression khác nhau
    for compression in np.arange(0.2, 0.6, 0.1):
        try:
            clusterer = CURE(n_clusters=num_clusters, 
                           n_represent_points=n_represent, 
                           compression=compression)
            labels = clusterer.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            
            if score > best_score:
                best_score = score
                best_compression = compression
                best_labels = labels
        except:
            continue
    
    return best_labels, best_compression, best_score

def validate_csv_file(file):
    # Kiểm tra định dạng file
    if not file.filename.endswith('.csv'):
        raise ValueError("Chỉ chấp nhận file CSV")
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Kiểm tra các tham số đầu vào
    if 'file' not in request.files:
        return jsonify({"error": "Thiếu file dữ liệu"}), 400
    if 'fields' not in request.form:
        return jsonify({"error": "Thiếu trường dữ liệu"}), 400
    if 'numClusters' not in request.form:
        return jsonify({"error": "Thiếu số lượng nhóm"}), 400

    file = request.files['file']
    
    try:
        # Kiểm tra file
        validate_csv_file(file)
        
        # Phân tích và kiểm tra số lượng nhóm
        try:
            num_clusters = int(request.form['numClusters'])
            if num_clusters < 1 or num_clusters > 10:
                return jsonify({"error": "Số nhóm phải từ 1 đến 10"}), 400
        except ValueError:
            return jsonify({"error": "Số nhóm phải là số nguyên"}), 400

        # Phân tích các trường dữ liệu
        try:
            fields = json.loads(request.form['fields'])
        except json.JSONDecodeError:
            return jsonify({"error": "Định dạng trường dữ liệu không hợp lệ"}), 400

        # Đọc và kiểm tra DataFrame
        try:
            df = pd.read_csv(file)
            if df.empty:
                return jsonify({"error": "File CSV rỗng"}), 400
        except Exception as e:
            return jsonify({"error": f"Lỗi đọc file CSV: {str(e)}"}), 400

        if not fields:
            # Nếu không chọn trường, lấy tất cả cột số
            fields = df.select_dtypes(include=[np.number]).columns.tolist()
            if not fields:
                return jsonify({"error": "Không tìm thấy cột số trong dữ liệu"}), 400

        # Kiểm tra các trường được chọn có tồn tại trong DataFrame
        missing_fields = [f for f in fields if f not in df.columns]
        if missing_fields:
            return jsonify({"error": f"Các trường không tồn tại: {', '.join(missing_fields)}"}), 400

        # Tiền xử lý dữ liệu
        X_scaled, df_clean = preprocess_data(df, fields)
        
        if CURE is not None:
            n_samples = len(X_scaled)
            n_represent = min(max(int(np.log2(n_samples) * 4), 10), 50)
            
            # Tối ưu hóa phân cụm
            cluster_labels, best_compression, silhouette = optimize_clustering(X_scaled, num_clusters, n_represent)
            
            # Nếu tối ưu hóa thất bại, sử dụng tham số mặc định
            if cluster_labels is None:
                compression = 0.5 - (np.log2(n_samples) - 5) * 0.03
                compression = min(max(compression, 0.2), 0.5)
                clusterer = CURE(n_clusters=num_clusters, n_represent_points=n_represent, compression=compression)
                cluster_labels = clusterer.fit_predict(X_scaled)
        else:
            # Sử dụng KMeans nếu không có CURE
            clusterer = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(X_scaled)

        # PCA để giảm chiều dữ liệu xuống tối đa 3 chiều để vẽ
        n_components = min(3, len(fields))
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Chuẩn bị thống kê cho từng cụm
        cluster_stats = {}
        for i in range(num_clusters):
            cluster_mask = cluster_labels == i
            cluster_data = df_clean[fields].iloc[cluster_mask]
            cluster_stats[f"Nhóm {i}"] = {
                "size": int(cluster_mask.sum()),
                "mean": cluster_data.mean().to_dict(),
                "std": cluster_data.std().to_dict()
            }

        # Định dạng dữ liệu để hiển thị
        original_data = []
        for i in range(len(df_clean)):
            row_dict = {}
            for j, field in enumerate(fields[:n_components]):
                row_dict[field] = float(df_clean[field].iloc[i])
            original_data.append(row_dict)

        pca_data = []
        important_features = []
        for component in pca.components_:
            idx = np.argmax(np.abs(component))
            important_features.append(fields[idx])
        new_column_names = [f"PC{i+1} ({important_features[i]})" for i in range(len(important_features))]
        
        for i in range(len(X_pca)):
            row_dict = {}
            for j, col in enumerate(new_column_names):
                row_dict[col] = float(X_pca[i][j])
            pca_data.append(row_dict)

        cluster_labels_python = [int(label) for label in cluster_labels.tolist()]

        return jsonify({
            "original_data": original_data,
            "pca": pca_data,
            "labels": {
                "original": fields[:n_components],
                "pca": new_column_names
            },
            "cluster_labels": cluster_labels_python,
            "groups": [f"Khách {i+1}" for i in range(len(pca_data))],
            "cluster_stats": cluster_stats,
            "silhouette_score": float(silhouette) if 'silhouette' in locals() else None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if 'message' not in data or 'cluster_stats' not in data:
            return jsonify({"error": "Thiếu thông tin"}), 400

        message = data['message'].lower()
        stats = data['cluster_stats']
        
        # Phân tích các cụm dựa trên truy vấn của người dùng
        response = analyze_clusters_ai(message, stats)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def analyze_clusters_ai(message, stats):
    # So khớp mẫu 
    if any(word in message for word in ["so sánh", "khác nhau", "khác biệt"]):
        return compare_clusters(stats)
    
    elif any(word in message for word in ["đặc trưng", "nổi bật", "đặc điểm chính"]):
        return get_key_characteristics(stats)
    
    elif any(word in message for word in ["xu hướng", "pattern", "mẫu"]):
        return identify_patterns(stats)
    
    elif any(word in message for word in ["số lượng", "kích thước", "size"]):
        return analyze_cluster_sizes(stats)
    
    elif any(word in message for word in ["trung bình", "mean", "average"]):
        return analyze_cluster_means(stats)
    
    elif any(word in message for word in ["phân phối", "distribution"]):
        return analyze_distributions(stats)
    
    else:
        return general_cluster_analysis(stats)

def compare_clusters(stats):
    response = "So sánh giữa các nhóm:\n\n"
    for field in next(iter(stats.values()))["mean"].keys():
        response += f"* {field}:\n"
        values = {group: data["mean"][field] for group, data in stats.items()}
        max_group = max(values.items(), key=lambda x: x[1])
        min_group = min(values.items(), key=lambda x: x[1])
        response += f"  - Cao nhất ở {max_group[0]}: {max_group[1]:.2f}\n"
        response += f"  - Thấp nhất ở {min_group[0]}: {min_group[1]:.2f}\n"
    return response

def get_key_characteristics(stats):
    response = "Đặc điểm nổi bật của từng nhóm:\n\n"
    for group, data in stats.items():
        response += f"{group}:\n"
        means = data["mean"]
        stds = data["std"]
        # Tìm những đặc điểm nổi bật nhất
        distinctive = []
        for field in means:
            other_means = [s["mean"][field] for g, s in stats.items() if g != group]
            if other_means:
                avg_others = sum(other_means) / len(other_means)
                z_score = abs(means[field] - avg_others) / (stds[field] if stds[field] > 0 else 1)
                distinctive.append((field, z_score, means[field] > avg_others))
        
        distinctive.sort(key=lambda x: x[1], reverse=True)
        for field, score, is_higher in distinctive[:3]:
            comparison = "cao hơn" if is_higher else "thấp hơn"
            response += f"  - {field}: {comparison} đáng kể ({means[field]:.2f})\n"
    return response

def identify_patterns(stats):
    response = "Các xu hướng và mẫu trong dữ liệu:\n\n"
    
    # Phân tích mối tương quan
    for group, data in stats.items():
        response += f"{group}:\n"
        means = data["mean"]
        # Tìm các đặc điểm
        features = list(means.keys())
        patterns = []
        for i, f1 in enumerate(features):
            for f2 in features[i+1:]:
                if abs(means[f1] - means[f2]) < 0.5:  # Giá trị tương tự
                    patterns.append(f"{f1} và {f2} có giá trị tương đương")
        if patterns:
            response += "  Các đặc điểm liên quan:\n"
            for pattern in patterns[:3]:  # Hiển thị 3 mẫu ảnh hưởng
                response += f"  - {pattern}\n"
    return response

def analyze_cluster_sizes(stats):
    total = sum(data["size"] for data in stats.values())
    response = "Phân tích kích thước nhóm:\n\n"
    for group, data in stats.items():
        size = data["size"]
        percentage = (size / total) * 100
        response += f"{group}:\n"
        response += f"  - Số lượng: {size} khách hàng\n"
        response += f"  - Tỷ lệ: {percentage:.1f}%\n"
    return response

def analyze_cluster_means(stats):
    response = "Giá trị trung bình của các nhóm:\n\n"
    for group, data in stats.items():
        response += f"{group}:\n"
        for field, value in data["mean"].items():
            std = data["std"][field]
            response += f"  - {field}: {value:.2f} (±{std:.2f})\n"
    return response

def analyze_distributions(stats):
    response = "Phân tích phân phối các nhóm:\n\n"
    for group, data in stats.items():
        response += f"{group}:\n"
        means = data["mean"]
        stds = data["std"]
        for field in means:
            cv = (stds[field] / means[field]) * 100 if means[field] != 0 else 0
            if cv < 20:
                spread = "tập trung"
            elif cv < 50:
                spread = "phân tán vừa phải"
            else:
                spread = "phân tán rộng"
            response += f"  - {field}: {spread} (CV={cv:.1f}%)\n"
    return response

def general_cluster_analysis(stats):
    response = "Tổng quan về các nhóm khách hàng:\n\n"
    
    # phân tích kích thước tổng thể
    total = sum(data["size"] for data in stats.values())
    largest = max(stats.items(), key=lambda x: x[1]["size"])
    smallest = min(stats.items(), key=lambda x: x[1]["size"])
    
    response += f"Tổng số khách hàng: {total}\n"
    response += f"Nhóm lớn nhất: {largest[0]} ({largest[1]['size']} khách hàng)\n"
    response += f"Nhóm nhỏ nhất: {smallest[0]} ({smallest[1]['size']} khách hàng)\n\n"
    
    # Đặc điểm chính của từng nhóm
    response += "Đặc điểm chính của từng nhóm:\n"
    for group, data in stats.items():
        response += f"\n{group}:\n"
        means = data["mean"]
        # Tìm giá trị cực đại nhất
        sorted_features = sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, value in sorted_features[:3]:
            response += f"  - {feature}: {value:.2f}\n"
    
    return response

if __name__ == '__main__':
    app.run(debug=True)
