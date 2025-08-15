# 🩺 Diabetes Prediction System

A comprehensive Streamlit web application for diabetes prediction using machine learning. This app provides both single patient predictions and batch processing capabilities through CSV file uploads.

## 📋 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Model Information](#model-information)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 🔍 Single Patient Prediction
- Interactive form for individual patient data entry
- Real-time prediction results with visual feedback
- Input validation and error handling

### 📂 Batch Processing
- CSV file upload for multiple patient predictions
- Automatic data validation and preprocessing
- Results export with downloadable CSV
- Summary statistics and visualizations

### 🎯 Model Training
- Train new models with custom datasets
- Support for uploading training data
- Model performance metrics display
- Data standardization and preprocessing

### 📊 Additional Features
- Sample CSV template download
- Responsive web interface
- Medical disclaimer and privacy notes
- Real-time accuracy metrics

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Required Packages
```bash
pip install streamlit pandas numpy scikit-learn
```

### Clone/Download
```bash
git clone <repository-url>
cd diabetes-prediction-app
```

## 💻 Usage

### 1. Start the Application
```bash
streamlit run diabetes_app.py
```

### 2. Access the Web Interface
- Open your browser and go to `http://localhost:8501`
- The application will load with the main dashboard

### 3. Train the Model
**Option A: Use Default Data**
- Click "Train New Model" in the sidebar
- The app will use built-in sample data

**Option B: Upload Training Data**
- Click "Upload Training Dataset" in the sidebar
- Upload your diabetes.csv file (must include 'Outcome' column)
- Click "Train New Model"

**Option C: Use Local File**
- Place your `diabetes.csv` file in the same directory as the app
- Restart the application
- Click "Train New Model"

### 4. Make Predictions

**Single Prediction:**
1. Fill in the patient information form
2. Click "🔍 Predict"
3. View the result and recommendation

**Batch Prediction:**
1. Upload a CSV file with patient data
2. Ensure all required columns are present
3. Click "🔍 Run Batch Prediction"
4. Download results as CSV

## 📊 Data Requirements

### Required Columns for Prediction
Your CSV file must contain these exact column names:

| Column Name | Description | Data Type | Range |
|-------------|-------------|-----------|-------|
| `Pregnancies` | Number of pregnancies | Integer | 0-20 |
| `Glucose` | Plasma glucose concentration | Float | 0-300 |
| `BloodPressure` | Diastolic blood pressure (mm Hg) | Float | 0-200 |
| `SkinThickness` | Triceps skin fold thickness (mm) | Float | 0-100 |
| `Insulin` | 2-Hour serum insulin (mu U/ml) | Float | 0-1000 |
| `BMI` | Body mass index | Float | 0-70 |
| `DiabetesPedigreeFunction` | Diabetes pedigree function | Float | 0.078-3.0 |
| `Age` | Age in years | Integer | 21-120 |

### Training Data Requirements
For training, your dataset must also include:
- `Outcome`: Target variable (0 = Non-Diabetic, 1 = Diabetic)

### Sample Data Format
```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
6,148,72,35,0,33.6,0.627,50
1,85,66,29,0,26.6,0.351,31
8,183,64,0,0,23.3,0.672,32
```

## 🤖 Model Information

### Algorithm
- **Support Vector Machine (SVM)** with linear kernel
- Optimized for binary classification (Diabetic vs Non-Diabetic)

### Preprocessing
- **StandardScaler**: All features are standardized to zero mean and unit variance
- **Missing Value Handling**: Automatic filling with column means
- **Data Validation**: Automatic column name matching and validation

### Performance Metrics
- Training and test accuracy displayed in real-time
- Model performance shown in sidebar after training

### Model Architecture
```
Input Features (8) → StandardScaler → SVM (Linear Kernel) → Binary Output (0/1)
```

## 📁 File Structure

```
diabetes-prediction-app/
├── diabetes_app.py          # Main Streamlit application
├── diabetes.csv             # Training dataset (optional)
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── sample_predictions/      # Example output files
    └── sample_results.csv
```

## 🔧 Troubleshooting

### Common Issues

**1. "No diabetic predictions" Error**
- **Cause**: Model trained on incompatible data
- **Solution**: Upload your actual diabetes training dataset or place `diabetes.csv` in app directory

**2. "Missing columns" Error**
- **Cause**: CSV file doesn't have required column names
- **Solution**: Ensure exact column names match requirements (case-sensitive)

**3. "Model not found" Error**
- **Cause**: Model hasn't been trained yet
- **Solution**: Click "Train New Model" in the sidebar first

**4. Import Errors**
- **Cause**: Missing required packages
- **Solution**: Run `pip install -r requirements.txt`

### File Format Issues
- Ensure CSV files use comma (`,`) as delimiter
- Remove any extra spaces in column names
- Check for special characters in data values
- Verify numeric columns contain only numbers

### Performance Issues
- Large CSV files (>10MB) may take longer to process
- Consider splitting very large datasets into smaller batches
- Restart the app if memory usage becomes high

## 📈 Example Workflows

### Workflow 1: Quick Single Prediction
1. Start app → Train model → Enter patient data → Get result

### Workflow 2: Batch Processing
1. Start app → Train model → Upload CSV → Download results

### Workflow 3: Custom Training
1. Start app → Upload training data → Train model → Make predictions

## 🏥 Medical Disclaimer

⚠️ **Important Medical Disclaimer:**
- This application is for **educational and research purposes only**
- Results should **NOT replace professional medical diagnosis**
- Always consult qualified healthcare professionals for medical decisions
- The model's predictions may not be 100% accurate

## 🔒 Privacy & Security

- **Data Privacy**: Uploaded files are processed locally and not stored permanently
- **No Data Storage**: Patient information is not saved or transmitted
- **Local Processing**: All computations happen on your local machine

## 🤝 Contributing

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Improvement
- Additional ML algorithms (Random Forest, Logistic Regression)
- Data visualization and analytics
- Model interpretability features
- Enhanced UI/UX design
- API integration capabilities

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or suggestions:
- Create an issue in the repository
- Contact the development team
- Check the troubleshooting section above

## 🏷️ Version History

- **v1.0.0**: Initial release with basic prediction functionality
- **v1.1.0**: Added batch processing and CSV upload
- **v1.2.0**: Enhanced training data upload and model persistence
- **v1.3.0**: Improved UI/UX and error handling

---

**Built with ❤️ using Streamlit and scikit-learn**
Python App Setup and Run Guide
==============================

📌 Requirements
---------------
- Python 3.x installed
- Anaconda / Miniconda (optional but recommended)
- PowerShell with script execution enabled (`RemoteSigned` policy)

1. Clone or Download the Project
---------------------------------
git clone https://github.com/your-username/your-repo.git
cd your-repo

2. (Optional) Create and Activate Conda Environment
---------------------------------------------------
If you are using conda:
conda create --name myenv python=3.11 -y
conda activate myenv

3. Install Dependencies
-----------------------
If your project has a `requirements.txt` file:
pip install -r requirements.txt

4. Running the Python App
-------------------------
Method 1 — Basic:
python app.py

Method 2 — With Arguments:
python app.py arg1 arg2

5. Fixing "conda not recognized" in PowerShell
----------------------------------------------
If you see:
conda : The term 'conda' is not recognized...

Run this in Anaconda Prompt:
conda init powershell
Then restart PowerShell.

6. Fixing "Running Scripts is Disabled" Error
----------------------------------------------
If you see:
File C:\\Users\\<User>\\Documents\\WindowsPowerShell\\profile.ps1 cannot be loaded...

Run:
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
Then restart PowerShell.

7. Troubleshooting
------------------
Check Python version:
python --version

List Conda environments:
conda env list

Install a missing package:
pip install package-name

📜 License
----------
This project is licensed under the MIT License. Feel free to modify and share.
"""
