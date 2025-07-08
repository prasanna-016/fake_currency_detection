# Fake Currency Detection

This project is a machine learning-powered web application that detects fake currency notes from images.  
It uses Python, Flask, and deep learning models to identify whether a given currency note is real or fake.

---

## 🚀 Features

- Upload an image of a currency note for verification
- Predict whether the note is fake or real using a trained model
- Simple and user-friendly web interface (Flask + HTML/CSS)
- Displays prediction results along with model confidence

---

## 🛠 Technology Stack

- **Python**
- **Flask**
- **TensorFlow / Keras**
- **OpenCV**
- **HTML / CSS / Bootstrap**
- **Chart.js** (optional for charts)

---

## 📂 Project Structure

```text
fake_currency_detection/
├── app.py                     # Main Flask application
├── fake_currency_detector.ipynb  # Model training notebook
├── templates/
│   └── index.html             # Frontend template
├── static/                    # Static files (CSS, JS, images)
├── uploads/                   # Uploaded images (runtime)
├── dataset/                   # Training/testing dataset
├── confusion.png              # Confusion matrix image (optional)
├── curves.png                 # Model performance curve (optional)
└── README.md                  # Project readme


---

## ⚡ How to Run Locally

1️⃣ Clone this repository:
```bash
git clone https://github.com/prasanna-016/fake_currency_detection.git
cd fake_currency_detection
2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
3️⃣ Run the app:
```bash
python app.py
4️⃣ Open your browser and go to:
```bash
http://127.0.0.1:5000/

```
📝 Dataset
Original currency notes: Mendeley Indian Currency Dataset

Fake currency notes: Collected from Google and other public sources

🌟 Future Enhancements
Support more denominations

Improve model accuracy with more data

Add API support for mobile apps

Deploy on cloud platforms (Heroku, AWS)
