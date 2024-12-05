from flask import Flask, request, jsonify, Response
from predict import Predictor
from collections import OrderedDict
import json
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

predictor = Predictor();

def predict_with_restnetV1(image_path):
    return predictor.predictResnetV1(image_path)


@app.route('/plant-disease/restnet-v1/predict', methods=['POST'])
def prediksi_gambar():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        result = predict_with_restnetV1(file_path)
        
        result = OrderedDict([
            ("nama_tanaman", result["nama_tanaman"]),
            ("nama_penyakit", result["nama_penyakit"]),
            ("deskripsi_penyakit", result["deskripsi_penyakit"]),
            ("penanganan_penyakit", result["penanganan_penyakit"]),
            ("pencegahan_penyakit", result["pencegahan_penyakit"]),
            ("poin", result["point"]),
        ])
        
        response_json = json.dumps(result, ensure_ascii=False)
        return Response(response_json, content_type='application/json')
        
        return jsonify(result)

    return jsonify({"error": "File upload failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)
