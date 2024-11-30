import os
import numpy as np
from flask import Flask, request, jsonify
from pydub import AudioSegment
import tempfile
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import librosa

app = Flask(__name__)

# Fungsi Konversi dan Ekstraksi Fitur
def convert_to_temp_wav(audio_file):
    """Konversi audio ke file WAV sementara."""
    audio = AudioSegment.from_file(audio_file)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_file.name, format="wav")
    temp_file.close()
    return temp_file.name

def mfcc_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=28)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting MFCC features: {e}")
        return None

def chroma_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        return np.mean(chroma.T, axis=0)
    except Exception as e:
        print(f"Error extracting Chroma features: {e}")
        return None

def combined_features(file_path):
    mfcc = mfcc_features(file_path)
    chroma = chroma_features(file_path)
    
    if mfcc is None or chroma is None:
        print("One of the feature extractions failed.")
        return None
    
    print(f"MFCC Shape: {mfcc.shape if mfcc is not None else 'None'}")
    print(f"Chroma Shape: {chroma.shape if chroma is not None else 'None'}")
    
    return np.concatenate((mfcc, chroma))

def predict_song_genre(model, file_path, label_encoder):
    features = combined_features(file_path)
    
    if features is not None:
        pred_features = features.reshape(1, -1)
        pred = model.predict(pred_features)
        pred_class = np.argmax(pred)  # Indeks prediksi
        
        predicted_label = label_encoder.classes_[pred_class]
        pred_prob = pred[0][pred_class]
        
        return predicted_label, pred_prob
    else:
        return None

# Load Model dan Label Encoder
model = load_model("best_model.keras")
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["class1", "class2", "class3"])  # Ganti dengan label kelas Anda

# Endpoint untuk Prediksi
@app.route("/predict", methods=["POST"])
def predict():
    if "voice" not in request.files:
        return jsonify({"error": "No audio file found in the request"}), 400

    audio_file = request.files["voice"]

    try:
        # Konversi audio ke file WAV sementara
        temp_wav_path = convert_to_temp_wav(audio_file)
        result = predict_song_genre(model, temp_wav_path, label_encoder)
        
        # Hapus file sementara
        os.remove(temp_wav_path)
        
        if result:
            predicted_label, predicted_prob = result
            return jsonify({
                "predicted_class": predicted_label,
                "score": round(predicted_prob, 2)
            }), 200
        else:
            return jsonify({"error": "Failed to process audio file"}), 500

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

# Jalankan Flask Server
if __name__ == "__main__":
    app.run(debug=True)
