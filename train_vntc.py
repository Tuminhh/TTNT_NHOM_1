
import os
import pandas as pd
import gensim
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle
import glob

DATA_PATH = 'VNTC_Data/Train_Full' 

def load_data(path):
    data = []
    labels = []
    for label in os.listdir(path):
        label_dir = os.path.join(path, label)
        if os.path.isdir(label_dir):
            print(f"Đang đọc dữ liệu chủ đề: {label}...")
            for file_path in glob.glob(os.path.join(label_dir, '*.txt')):
                try:
                    with open(file_path, 'r', encoding='utf-16') as f:
                        content = f.read()
                        if content.strip(): 
                            data.append(content)
                            labels.append(label)
                except Exception as e:
                    print(f"Lỗi đọc file {file_path}: {e}")
    return data, labels

# 1. Đọc dữ liệu
print("Bắt đầu đọc dữ liệu từ ổ cứng...")
texts, raw_labels = load_data(DATA_PATH)
df = pd.DataFrame({'text': texts, 'label': raw_labels})
print(f"Tổng số bài báo đã đọc: {len(df)}")

# 2. Tiền xử lý 
def preprocessing(text):
    # Gensim: xóa dấu, ký tự đặc biệt, chuyển chữ thường
    text_list = gensim.utils.simple_preprocess(text)
    text = ' '.join(text_list)
    # PyVi: Tách từ tiếng Việt 
    text = ViTokenizer.tokenize(text)
    return text

print("Đang tiền xử lý văn bản (có thể mất vài phút)...")
df['clean_text'] = df['text'].apply(preprocessing)

# 3. Mã hóa nhãn 
encoder = LabelEncoder()
y_data = encoder.fit_transform(df['label'])

# 4. TF-IDF 
print("Đang tạo vector TF-IDF (max_features=30000)...")
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
X_tfidf = tfidf_vect.fit_transform(df['clean_text'])

# 5. Giảm chiều SVD 
print("Đang giảm chiều dữ liệu bằng SVD (300 components)...")
svd = TruncatedSVD(n_components=300, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

# 6. Huấn luyện SVM 
print("Đang huấn luyện mô hình SVM...")
svm_model = SVC()
svm_model.fit(X_svd, y_data)

# 7. Lưu tất cả components
print("Đang lưu mô hình vào file 'vntc_full_model.pkl'...")
with open('vntc_full_model.pkl', 'wb') as f:
    pickle.dump({
        'encoder': encoder,
        'tfidf': tfidf_vect,
        'svd': svd,
        'model': svm_model
    }, f)

print("Hoàn tất! Hãy chạy app.py")