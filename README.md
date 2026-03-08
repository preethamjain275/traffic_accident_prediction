<div align="center">

<!-- Animated Header -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=TrafficGuard%20AI&fontSize=70&fontColor=fff&animation=twinkling&fontAlignY=35&desc=AI%20Powered%20Road%20Safety%20Analytics&descAlignY=55&descSize=20"/>

<!-- Typing Animation -->
<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=22&pause=1000&color=F97316&center=true&vCenter=true&width=700&height=80&lines=Predicting+Road+Accident+Severity;Powered+by+Random+Forest+Algorithm;Real-Time+Interactive+Dashboard;22+Feature+Weather+%26+Road+Analysis" alt="Typing SVG" />
</a>

<br/>

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ScikitLearn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

<br/>

![Status](https://img.shields.io/badge/Status-Active-22c55e?style=flat-square)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-62.35%25-f97316?style=flat-square)
![Trees](https://img.shields.io/badge/Decision%20Trees-200-ef4444?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-10%2C000%20Records-3b82f6?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)
![Made With Love](https://img.shields.io/badge/Made%20with-❤️-red?style=flat-square)

</div>

---

<div align="center">

## 🎬 Live Preview

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700"/>

</div>

---

## 📌 Table of Contents

- [About The Project](#-about-the-project)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Algorithm](#-algorithm---random-forest)
- [App Pages](#-app-pages)
- [Installation](#-installation--setup)
- [How to Run](#-how-to-run)
- [Model Performance](#-model-performance)
- [Screenshots](#-screenshots)

---

## 🚀 About The Project

<img align="right" width="380" src="https://user-images.githubusercontent.com/74038190/229223263-cf2e4b07-2615-4f87-9c38-e37600f8381a.gif"/>

**TrafficGuard AI** is a Traffic Accident Prediction System that uses **Machine Learning** to predict the severity of road accidents in real time.

By analyzing **22 input features** — including weather conditions, road type, speed limits, and time of day — the system classifies accident severity into 4 levels:

| Level | Severity | Description |
|:---:|:---:|:---|
| 🟢 **1** | LOW | Minor incident, minimal disruption |
| 🟡 **2** | MODERATE | Significant delays expected |
| 🟠 **3** | HIGH | Road likely blocked |
| 🔴 **4** | CRITICAL | Emergency response required |

The project features a fully interactive **Streamlit web dashboard** with dark cyberpunk aesthetics, real-time prediction, animated charts, and a risk radar visualization.

<br clear="right"/>

---

## ✨ Key Features

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="400"/>
</div>

```
🔮  Real-time accident severity prediction
📊  Interactive Plotly dashboards & charts
🎯  Risk Radar visualization per prediction
🌲  Random Forest with 200 decision trees
📈  5-Fold Cross Validation evaluation
🔥  Feature importance analysis
🗂️  Full dataset explorer with filters
🎨  Advanced dark UI with cyberpunk theme
⚡  Instant prediction with probability bars
🏆  Confusion matrix visualization
```

---

## 🛠 Tech Stack

<div align="center">

| Category | Technology | Purpose |
|:---:|:---:|:---|
| **Language** | ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white) | Core programming language |
| **ML Algorithm** | ![ScikitLearn](https://img.shields.io/badge/-Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Random Forest Classifier |
| **UI Framework** | ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | Interactive web dashboard |
| **Visualization** | ![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) | Charts, graphs, radar |
| **Data Handling** | ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white) | Dataset processing |
| **Math** | ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical computations |
| **Model Saving** | ![Joblib](https://img.shields.io/badge/-Joblib-gray?style=flat) | Save/load .pkl model files |

</div>

---

## 📁 Project Structure

```
🗂️ traffic_accident_prediction/
│
├── 📄 app.py                      ← Main Streamlit UI (4 pages)
├── 📄 generate_dataset.py         ← Generates 10,000 accident records
├── 📄 model_trainer.py            ← Trains & saves Random Forest model
├── 📄 requirements.txt            ← All Python dependencies
│
├── 📂 data/
│   └── 📊 accidents_cleaned.csv  ← Auto-generated cleaned dataset
│
└── 📂 models/
    ├── 🧠 rf_model.pkl            ← Trained Random Forest model
    ├── 🔤 le_weather.pkl          ← Label encoder — weather
    ├── 🔤 le_road.pkl             ← Label encoder — road type
    ├── 🔤 le_day.pkl              ← Label encoder — day of week
    ├── 🔤 le_state.pkl            ← Label encoder — US state
    └── 📋 model_meta.json         ← Accuracy + feature metadata
```

---

## 📊 Dataset

The dataset contains **10,000 realistic accident records** with the following features:

<details>
<summary><b>📋 Click to expand — All 19 Features</b></summary>

<br/>

| Feature | Type | Description |
|:---|:---:|:---|
| `Severity` | Target | 1 (Low) → 4 (Critical) |
| `Temperature_F` | Numeric | Temperature in Fahrenheit |
| `Wind_Speed_mph` | Numeric | Wind speed in mph |
| `Visibility_mi` | Numeric | Visibility in miles |
| `Precipitation_in` | Numeric | Rain/snow amount |
| `Humidity_pct` | Numeric | Humidity percentage |
| `Pressure_in` | Numeric | Atmospheric pressure |
| `Speed_Limit` | Numeric | Road speed limit |
| `Weather_Condition` | Categorical | Clear/Rain/Snow/Fog etc. |
| `Road_Type` | Categorical | Highway/City Street etc. |
| `Hour` | Numeric | Hour of accident (0–23) |
| `Day_of_Week` | Categorical | Monday to Sunday |
| `Month` | Numeric | 1–12 |
| `State` | Categorical | US State abbreviation |
| `Junction` | Binary | Junction present? |
| `Traffic_Signal` | Binary | Signal present? |
| `Crossing` | Binary | Pedestrian crossing? |
| `Stop` | Binary | Stop sign present? |
| `Amenity` | Binary | Amenity nearby? |

</details>

---

## 🌲 Algorithm — Random Forest

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212748842-9fcbad5b-6173-4175-8a61-521f3dbb7514.gif" width="500"/>
</div>

> **Simple Analogy:** Imagine asking 200 experts the same question. Each gives their answer. The majority vote becomes the final prediction. That's Random Forest.

```python
RandomForestClassifier(
    n_estimators   = 200,      # 200 decision trees
    max_depth      = 15,       # tree depth limit
    max_features   = 'sqrt',   # features per split
    class_weight   = 'balanced', # handles imbalanced data
    random_state   = 42
)
```

**Why Random Forest?**
- ✅ Handles both numerical and categorical data
- ✅ Works well with imbalanced class distribution
- ✅ Provides feature importance rankings
- ✅ Resistant to overfitting
- ✅ No need for feature scaling

---

## 🖥️ App Pages

<div align="center">

| Page | Description |
|:---:|:---|
| 🏠 **Dashboard** | KPI cards, severity distribution, hourly trends, weather heatmap |
| 🔮 **Predict Severity** | Input conditions → live severity prediction + probability bars + radar chart |
| 📊 **Model Analytics** | Feature importance, confusion matrix, model parameters |
| 🗂️ **Data Explorer** | Browse, filter, visualize the full dataset |

</div>

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- VS Code (recommended)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/traffic_accident_prediction.git
cd traffic_accident_prediction
```

### Step 2 — Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
# Step 1 — Generate dataset
python generate_dataset.py

# Step 2 — Train the model
python model_trainer.py

# Step 3 — Launch the app
python -m streamlit run app.py
```

App opens at → **http://localhost:8501** 🚀

> Next time, only run Steps 2 commands — dataset and model are already saved!

---

## 📈 Model Performance

<div align="center">

| Metric | Score |
|:---:|:---:|
| **Test Accuracy** | 62.35% |
| **CV Mean (5-Fold)** | 61.81% |
| **CV Std Dev** | ±0.38% |
| **Training Size** | 8,000 records |
| **Test Size** | 2,000 records |

</div>

### 🏆 Top Features by Importance

```
Bad Weather      ████████████████░░░░  13.6%
Speed Limit      ████████████████░░░░  13.6%
Visibility       ████████████░░░░░░░░  10.3%
Wind Speed       █████████░░░░░░░░░░░   7.9%
Weather Type     █████████░░░░░░░░░░░   7.7%
Temperature      ████████░░░░░░░░░░░░   7.1%
```

---

## 👨‍💻 Author

<div align="center">

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif" width="400"/>

### **Preetham Jain M**

![Author](https://img.shields.io/badge/Author-Preetham%20Jain%20M-f97316?style=for-the-badge&logo=github&logoColor=white)

</div>

---

## 📜 License

<div align="center">

```
MIT License

Copyright (c) 2026 Preetham Jain M

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

![License](https://img.shields.io/badge/License-MIT-f97316?style=for-the-badge)

</div>

---

<div align="center">

<!-- Footer Wave -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer"/>

**⭐ Star this repo if you found it helpful!**

</div>
