CURE - Phân cụm dữ liệu với thuật toán CURE

Mô tả

  Dự án này áp dụng thuật toán CURE (Clustering Using Representatives) để phân cụm dữ liệu khách hàng. Hệ thống có khả năng trực quan hóa dữ liệu gốc và dữ liệu sau khi giảm chiều bằng PCA, từ đó giúp người dùng phân tích đặc điểm từng nhóm khách hàng một cách trực quan và hiệu quả.
  
  Ứng dụng được xây dựng bằng Python (Flask), HTML, JavaScript và sử dụng thư viện Plotly để trực quan hóa.

Cấu trúc thư mục

CURE/
├── dulieu/                     # Thư mục chứa dữ liệu người dùng tải lên
├── static/
│   └── script.js               # Mã JavaScript xử lý frontend và vẽ biểu đồ
├── templates/
│   ├── index.html              # Giao diện chính
│   └── result.html             # Trang hiển thị kết quả
├── venv/                       # Môi trường ảo Python
├── app.py                      # File chính chạy Flask server
└── taokhachhang_muasam.py      # Script tạo dữ liệu khách hàng mẫu


Chức năng chính

  Tải file CSV chứa dữ liệu khách hàng
  
  Chọn trường số để phân tích (2-3 trường)
  
  Chọn số lượng cụm cần phân chia (1–10 cụm)
  
  Thực hiện phân cụm bằng thuật toán CURE
  
  Trực quan hóa dữ liệu gốc và dữ liệu sau PCA bằng biểu đồ 3D
  
  Hệ thống gợi ý phân tích đặc điểm từng cụm

Hướng dẫn sử dụng

Cài đặt môi trường

  pip install flask pandas numpy scikit-learn plotly
