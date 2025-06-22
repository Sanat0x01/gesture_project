
# 👋 Gesture Recognition System using MediaPipe & KNN

This project implements a real-time hand gesture recognition system using **MediaPipe** for hand tracking and **K-Nearest Neighbors (KNN)** for classification. It maps common gestures to system actions like controlling volume, media, or switching windows.

---

## 📌 Features

- Real-time hand tracking using MediaPipe
- Gesture recording and custom dataset generation
- Training a gesture classification model (KNN)
- Live webcam-based gesture prediction
- Mapped actions:
  - 👍 Thumbs Up → Volume Up
  - 👎 Thumbs Down → Volume Down
  - ✊ Fist → Switch Window
  - ✋ Palm → Play/Pause

---

## 📂 Project Structure

```
gesture_project/
├── collect_gestures.py      # For recording gesture data
├── extract_landmarks.py     # Extracts 63 landmark features
├── train_model.py           # Trains and saves KNN model
├── predict_gesture.py       # Runs live webcam prediction
├── webcam_test.py           # (Optional) test live tracking
├── hand_tracking.py         # MediaPipe wrapper
├── data/                    # Stores CSVs of gestures
├── venv/                    # Python virtual environment (ignored)
└── gesture_knn_model.pkl    # Trained model (generated after training)
```

---

## ⚙️ Requirements

Install Python dependencies using:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install mediapipe opencv-python scikit-learn numpy
```

---

## 🚀 How to Run

### 1. Activate virtual environment (if using one)

```bash
source venv/bin/activate
```

### 2. Record gestures

```bash
python collect_gestures.py
```

Follow on-screen prompts to label and record samples.

### 3. Train the model

```bash
python train_model.py
```

It will save the model as `gesture_knn_model.pkl`.

### 4. Predict gestures live

```bash
python predict_gesture.py
```

Performs real-time prediction and system-level action mapping.

---

## 🧠 How It Works

- MediaPipe extracts 21 hand landmarks (x, y, z)
- These 63 values are used as features for KNN
- The model is trained on labeled CSV data
- During live webcam use, predicted gestures trigger mapped system actions

---

## 🧪 Test Webcam Feed (Optional)

```bash
python webcam_test.py
```

This just shows landmarks and hand detection without classification.

---

## 🧾 License

MIT License — feel free to use, modify, or contribute.

---

## 🤝 Contributing

Feel free to fork the repo, open issues or PRs.  
New gesture ideas or action integrations are welcome!

---

## 🔗 Author

**Sanat0x01**  
[GitHub Profile →](https://github.com/Sanat0x01)
