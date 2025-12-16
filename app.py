
from flask import Flask, render_template, request
import pickle
import gensim
from pyvi import ViTokenizer

app = Flask(__name__)

# Load mô hình lớn
try:
    with open('vntc_full_model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        encoder = saved_data['encoder']
        tfidf_vect = saved_data['tfidf']
        svd = saved_data['svd']
        model = saved_data['model']
    print("Đã tải mô hình VNTC thành công!")
except FileNotFoundError:
    print("CHƯA CÓ MÔ HÌNH: Hãy chạy file train_vntc.py trước để tạo file .pkl")

def preprocessing(text):
    text = gensim.utils.simple_preprocess(text)
    text = ' '.join(text)
    text = ViTokenizer.tokenize(text)
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_label = ""
    input_text = ""
    
    if request.method == 'POST':
        input_text = request.form['content']
        if input_text:
            try:
                # 1. Tiền xử lý
                clean_text = preprocessing(input_text)
                # 2. Vector hóa
                vector = tfidf_vect.transform([clean_text])
                # 3. Giảm chiều SVD [cite: 283]
                vector_svd = svd.transform(vector)
                # 4. Dự đoán
                pred_index = model.predict(vector_svd)[0]
                # 5. Lấy tên nhãn
                prediction_label = encoder.inverse_transform([pred_index])[0]
            except Exception as e:
                prediction_label = f"Lỗi xử lý: {str(e)}"

    return render_template('index.html', prediction=prediction_label, original_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)