from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import mysql.connector
import io
import base64
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

app = Flask(__name__)


model = tf.keras.models.load_model("cnn_model.h5")


db_config = {
    'user': 'root',
    'password': 'Harish07',
    'host': '127.0.0.1',
    'database': 'skin_lesion_segmentation'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

def apply_kmeans(image, clusters=2):
    flat_image = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    segmented = kmeans.fit_predict(flat_image)
    return segmented.reshape(128, 128)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    
    
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = plt.imread(io.BytesIO(file_bytes))
    image = np.resize(image, (128, 128, 3)) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]
    result = "Tumor Detected" if prediction > 0.5 else "No Tumor"

   
    segmented_image = apply_kmeans(image[0])
 
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("INSERT INTO uploads (filename, mask_path) VALUES (%s, %s)", (file.filename, result))
    connection.commit()
    cursor.close()
    connection.close()

    return jsonify({'message': result})

if __name__ == '__main__':
    app.run(debug=True)
