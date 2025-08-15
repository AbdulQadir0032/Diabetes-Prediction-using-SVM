import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import io

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #41506e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🩺 Diabetes Prediction System</h1>', unsafe_allow_html=True)

# Sidebar for model training and info
with st.sidebar:
    st.markdown("## 📊 Model Information")
    st.markdown("""
    **Features Required:**
    - Pregnancies
    - Glucose
    - BloodPressure
    - SkinThickness
    - Insulin
    - BMI
    - DiabetesPedigreeFunction
    - Age
    
    **Output:**
    - 0: Non-Diabetic
    - 1: Diabetic
    """)
    
    # Model training section
    st.markdown("## 🔧 Model Training")
    train_model = st.button("Train New Model", help="Click to train a new model with sample data")

@st.cache_data
def load_sample_data():
    """Load actual diabetes dataset"""
    try:
        # Try to load the actual diabetes dataset
        return pd.read_csv('diabetes.csv')
    except FileNotFoundError:
        st.error("diabetes.csv not found! Please place your diabetes.csv file in the same directory as this app.")
        # Fallback to creating more realistic sample data based on actual diabetes dataset statistics
        np.random.seed(42)
        
        n_samples = 768
        data = {
            'Pregnancies': np.random.poisson(3.8, n_samples),
            'Glucose': np.random.normal(120.9, 31.9, n_samples).clip(0, 199),
            'BloodPressure': np.random.normal(69.1, 19.4, n_samples).clip(0, 122), 
            'SkinThickness': np.random.gamma(2, 10, n_samples).clip(0, 99),
            'Insulin': np.random.gamma(1.2, 66, n_samples).clip(0, 846),
            'BMI': np.random.normal(31.9, 7.9, n_samples).clip(0, 67.1),
            'DiabetesPedigreeFunction': np.random.gamma(2, 0.24, n_samples).clip(0.078, 2.42),
            'Age': np.random.gamma(4, 8, n_samples).clip(21, 81).astype(int)
        }
        
        df = pd.DataFrame(data)
        
        # Create more realistic outcome based on risk factors
        risk_score = (
            (df['Glucose'] > 140) * 2 +
            (df['BMI'] > 30) * 1.5 +
            (df['Age'] > 50) * 1 +
            (df['Pregnancies'] > 5) * 0.5 +
            (df['DiabetesPedigreeFunction'] > 0.5) * 1
        )
        
        # Create outcome with higher probability for high-risk individuals
        df['Outcome'] = np.random.binomial(1, 1 / (1 + np.exp(-risk_score + 3)), n_samples)
        
        return df

@st.cache_resource
def train_model_func():
    """Train the diabetes prediction model"""
    # Load data
    diabetes_dataset = load_sample_data()
    
    # Separate features and target
    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    Y = diabetes_dataset['Outcome']
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.2, stratify=Y, random_state=2
    )
    
    # Train the model
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    
    # Calculate accuracy
    train_pred = classifier.predict(X_train)
    test_pred = classifier.predict(X_test)
    train_accuracy = accuracy_score(train_pred, Y_train)
    test_accuracy = accuracy_score(test_pred, Y_test)
    
    return classifier, scaler, train_accuracy, test_accuracy

# Train model if button is clicked or if model doesn't exist
if train_model or 'model' not in st.session_state:
    with st.spinner("Training model... Please wait."):
        classifier, scaler, train_acc, test_acc = train_model_func()
        st.session_state.model = classifier
        st.session_state.scaler = scaler
        st.session_state.train_accuracy = train_acc
        st.session_state.test_accuracy = test_acc
    
    st.sidebar.success("Model trained successfully!")
    st.sidebar.write(f"Training Accuracy: {train_acc:.3f}")
    st.sidebar.write(f"Test Accuracy: {test_acc:.3f}")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h2 class="sub-header">📝 Single Prediction</h2>', unsafe_allow_html=True)
    
    # Input form for single prediction
    with st.form("single_prediction"):
        st.markdown("Enter patient information:")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        
        with col_b:
            insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=80)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.001)
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
        
        submit_single = st.form_submit_button("🔍 Predict", use_container_width=True)
        
        if submit_single and 'model' in st.session_state:
            # Prepare input data
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                  insulin, bmi, dpf, age]])
            
            # Standardize input
            input_scaled = st.session_state.scaler.transform(input_data)
            
            # Make prediction
            prediction = st.session_state.model.predict(input_scaled)
            
            # Display result
            if prediction[0] == 0:
                st.markdown('<div class="success-box"><h3>✅ Result: Non-Diabetic</h3><p>The model predicts that this person is not diabetic.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box"><h3>⚠️ Result: Diabetic</h3><p>The model predicts that this person is diabetic. Please consult a healthcare professional.</p></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<h2 class="sub-header">📂 Batch Prediction (CSV Upload)</h2>', unsafe_allow_html=True)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=['csv'],
        help="Upload a CSV file with the required columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Display file info
            st.write(f"**File uploaded:** {uploaded_file.name}")
            st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Show first few rows
            st.write("**Preview of uploaded data:**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Expected columns (handle both cases)
            expected_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            
            # Check for case variations in column names
            df_columns_lower = [col.lower() for col in df.columns]
            expected_columns_lower = [col.lower() for col in expected_columns]
            
            # Map actual column names to expected names
            column_mapping = {}
            for expected in expected_columns:
                for actual in df.columns:
                    if expected.lower() == actual.lower():
                        column_mapping[actual] = expected
                        break
            
            # Rename columns to match expected format
            if column_mapping:
                df = df.rename(columns=column_mapping)
                st.info(f"Renamed columns: {column_mapping}")
            
            # Check if all required columns are present
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")
                st.info("Please ensure your CSV file contains all required columns.")
            else:
                # Make predictions button
                if st.button("🔍 Run Batch Prediction", use_container_width=True):
                    if 'model' in st.session_state:
                        with st.spinner("Making predictions..."):
                            # Prepare data for prediction
                            X_new = df[expected_columns]
                            
                            # Handle missing values (optional)
                            if X_new.isnull().any().any():
                                st.warning("Missing values detected. Filling with column means.")
                                X_new = X_new.fillna(X_new.mean())
                            
                            # Standardize the new data
                            X_new_scaled = st.session_state.scaler.transform(X_new)
                            
                            # Make predictions
                            predictions = st.session_state.model.predict(X_new_scaled)
                            
                            # Add predictions to dataframe
                            result_df = df.copy()
                            result_df['Prediction'] = predictions
                            result_df['Prediction_Label'] = result_df['Prediction'].map({0: 'Non-Diabetic', 1: 'Diabetic'})
                            
                            # Display results
                            st.success("Predictions completed!")
                            
                            # Summary statistics
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Total Predictions", len(predictions))
                            with col_b:
                                st.metric("Non-Diabetic", sum(predictions == 0))
                            with col_c:
                                st.metric("Diabetic", sum(predictions == 1))
                            
                            # Show results
                            st.write("**Prediction Results:**")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Download button for results
                            csv_buffer = io.StringIO()
                            result_df.to_csv(csv_buffer, index=False)
                            csv_string = csv_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_string,
                                file_name=f"diabetes_predictions_{uploaded_file.name}",
                                mime="text/csv",
                                use_container_width=True
                            )
                    else:
                        st.error("Please train the model first using the sidebar.")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Please ensure your file is a valid CSV format.")

# Download sample CSV template
st.markdown("---")
st.markdown('<h3 class="sub-header">📋 Sample CSV Template</h3>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.write("Download a sample CSV template to understand the required format:")
    
    # Create sample data
    sample_data = {
        'Pregnancies': [6, 1, 8],
        'Glucose': [148, 85, 183],
        'BloodPressure': [72, 66, 64],
        'SkinThickness': [35, 29, 0],
        'Insulin': [0, 0, 0],
        'BMI': [33.6, 26.6, 23.3],
        'DiabetesPedigreeFunction': [0.627, 0.351, 0.672],
        'Age': [50, 31, 32]
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)

with col2:
    # Convert sample data to CSV for download
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Download Sample CSV",
        data=csv_string,
        file_name="diabetes_sample_template.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h4>📝 Important Notes:</h4>
    <ul>
        <li><strong>Medical Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice.</li>
        <li><strong>Data Privacy:</strong> Uploaded files are processed locally and not stored permanently.</li>
        <li><strong>Model Accuracy:</strong> The model's predictions are based on the training data and may not be 100% accurate.</li>
    </ul>
</div>
""", unsafe_allow_html=True)