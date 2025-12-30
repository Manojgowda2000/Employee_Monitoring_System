import os
import json
import numpy as np
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
import cv2

def load_embeddings(json_path):
    """Load saved embeddings file (name -> embedding list)."""
    if not os.path.exists(json_path):
        return [], np.empty((0, 512), dtype=np.float32)
    with open(json_path, "r") as f:
        data = json.load(f)
    names = list(data.keys())
    embeddings = np.array(list(data.values()), dtype=np.float32)
    return names, embeddings


def preprocess_face(img, size=160):
    """
    Preprocess face image for InceptionResnet-like model.
    Input img is BGR (cv2). Output shape (1,3,size,size), float32.
    """
    try:
        face = cv2.resize(img, (size, size))
    except Exception:
        face = cv2.resize(img.copy(), (size, size))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5
    face = np.transpose(face, (2, 0, 1))
    return np.expand_dims(face.astype(np.float32), axis=0)


class FaceRecognizer:
    """
    ONNX-based face embedder + simple cosine similarity recognition.
    - onnx_path: path to onnx model that outputs embedding (1,512)
    - embeddings_json: path to JSON with stored embeddings {name: [512 floats]}
    """

    def __init__(self, onnx_path, embeddings_json, providers=None):
        self.onnx_path = onnx_path
        self.embeddings_json = embeddings_json
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.names, self.embeddings = load_embeddings(embeddings_json)

        # normalize stored embeddings
        if self.embeddings.size:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = self.embeddings / norms

    def get_embedding(self, face_img):
        blob = preprocess_face(face_img)
        out = self.session.run(None, {self.session.get_inputs()[0].name: blob})[0]
        emb = out.flatten().astype(np.float32)
        # normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    
    def recognize(self, face_img, threshold=0.6):
        if self.embeddings.shape[0] == 0:
            return "Unknown", 0.0

        emb = self.get_embedding(face_img)
        sims = cosine_similarity([emb], self.embeddings)[0]
        idx = np.argmax(sims)
        best_score = float(sims[idx])

        # --- FIX: reject negatives immediately ---
        if best_score < 0:
            return "Unknown", best_score

        # --- apply threshold ---
        if best_score >= threshold:
            return self.names[idx], best_score

        return "Unknown", best_score


    def add_embedding(self, name, face_img):
        """
        Add a new embedding to memory and persist to JSON.
        face_img: BGR image of face
        """
        emb = self.get_embedding(face_img).tolist()
        # load existing file (dict), update and write
        data = {}
        if os.path.exists(self.embeddings_json):
            with open(self.embeddings_json, "r") as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = {}
        data[name] = emb
        with open(self.embeddings_json, "w") as f:
            json.dump(data, f)
        # refresh in-memory
        self.names, self.embeddings = load_embeddings(self.embeddings_json)
        if self.embeddings.size:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = self.embeddings / norms
