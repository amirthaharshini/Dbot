# dbot_data.py - FINAL VERSION with 5 Features (C, R, T, L, V) and XGBoost Decoding
import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier 

# --- Constants ---
DATA_FILE = 'motor_fault_data.csv'
N_ROWS = 1500  

# Global containers for mapping (needed for prediction decoding)
GLOBAL_LABELS = [] 
GLOBAL_INVERSE_MAPPING = {}

# --- Data Generation Logic ---
def generate_motor_data(n_rows=N_ROWS):
    """Generates a synthetic dataset for motor fault prediction, now with 5 features."""
    if os.path.exists(DATA_FILE):
        print(f"Loading existing data from {DATA_FILE}...")
        return pd.read_csv(DATA_FILE)

    print(f"Generating {n_rows} rows of synthetic motor data (5 Features, 6 Fault Classes)...")
    np.random.seed(42)
    
    # Base operating conditions
    base_current = 20.0
    base_resistance = 5.0
    base_temperature = 55.0
    base_vibration = 0.5 
    
    data = {
        'Current (A)': np.abs(np.random.normal(base_current, 3, n_rows)),
        'Resistance (Ω)': np.abs(np.random.normal(base_resistance, 0.5, n_rows)),
        'Temperature (°C)': np.abs(np.random.normal(base_temperature, 5, n_rows)),
        'Load (%)': np.random.randint(0, 101, n_rows),
        'Vibration (g)': np.abs(np.random.normal(base_vibration, 0.1, n_rows)), # NEW FEATURE
        'Fault': ['Normal'] * n_rows
    }
    df = pd.DataFrame(data)
    
    fault_prob = 0.05
    
    for i in range(n_rows):
        r = np.random.rand()
        
        if r < fault_prob: # 1. Bearing Fault
            df.loc[i, 'Current (A)'] += np.random.uniform(3, 5)
            df.loc[i, 'Temperature (°C)'] += np.random.uniform(10, 20)
            df.loc[i, 'Vibration (g)'] += np.random.uniform(1.5, 3.0) 
            df.loc[i, 'Fault'] = 'Bearing Fault'
        elif r < 2 * fault_prob: # 2. Winding Short
            df.loc[i, 'Current (A)'] += np.random.uniform(5, 8)
            df.loc[i, 'Resistance (Ω)'] -= np.random.uniform(1, 2)
            df.loc[i, 'Vibration (g)'] += np.random.uniform(0.1, 0.5) 
            df.loc[i, 'Fault'] = 'Winding Short'
        elif r < 3 * fault_prob: # 3. Overload
            df.loc[i, 'Current (A)'] += np.random.uniform(8, 12)
            df.loc[i, 'Load (%)'] = np.random.randint(90, 101)
            df.loc[i, 'Vibration (g)'] += np.random.uniform(0.5, 1.0)
            df.loc[i, 'Fault'] = 'Overload'
        elif r < 4 * fault_prob: # 4. Misalignment/Imbalance
            df.loc[i, 'Current (A)'] += np.random.uniform(1.5, 3) 
            df.loc[i, 'Temperature (°C)'] += np.random.uniform(5, 10) 
            df.loc[i, 'Vibration (g)'] += np.random.uniform(0.8, 1.5)
            df.loc[i, 'Fault'] = 'Misalignment/Imbalance'
        elif r < 5 * fault_prob: # 5. Rotor Bar Damage
            df.loc[i, 'Current (A)'] += np.random.uniform(1, 2) 
            df.loc[i, 'Temperature (°C)'] += np.random.uniform(2, 5) 
            df.loc[i, 'Vibration (g)'] += np.random.uniform(0.3, 0.7) 
            df.loc[i, 'Fault'] = 'Rotor Bar Damage'
        
        # Clamp values to realistic ranges
        df.loc[i, 'Current (A)'] = round(max(0, df.loc[i, 'Current (A)']), 2)
        df.loc[i, 'Resistance (Ω)'] = round(max(1, df.loc[i, 'Resistance (Ω)']), 2)
        df.loc[i, 'Temperature (°C)'] = round(max(20, df.loc[i, 'Temperature (°C)']), 2)
        df.loc[i, 'Vibration (g)'] = round(max(0.1, df.loc[i, 'Vibration (g)']), 2)

    df.to_csv(DATA_FILE, index=False)
    return df

# --- ML Model Training and Prediction ---
def train_and_get_model():
    """Trains the XGBoost Classifier and returns the model and its accuracy."""
    global GLOBAL_LABELS, GLOBAL_INVERSE_MAPPING # Use globals to store mapping

    df = generate_motor_data()
        
    X = df[['Current (A)', 'Resistance (Ω)', 'Temperature (°C)', 'Load (%)', 'Vibration (g)']]
    y = df['Fault']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Setup Label Encoding
    labels = sorted(y.unique())
    label_mapping = {label: i for i, label in enumerate(labels)}
    y_train_encoded = y_train.map(label_mapping)
    y_test_encoded = y_test.map(label_mapping)
    
    # Store mapping globally for use in predict_fault
    GLOBAL_LABELS = labels
    GLOBAL_INVERSE_MAPPING = {i: label for label, i in label_mapping.items()}
    
    # 2. Train XGBClassifier
    model = XGBClassifier(
        objective='multi:softmax', 
        num_class=len(labels),
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='merror', 
        random_state=42
    )
    model.fit(X_train, y_train_encoded)
    
    y_pred_encoded = model.predict(X_test)
    
    # 3. Decode predictions back to original labels for reporting
    y_pred = pd.Series(y_pred_encoded).map(GLOBAL_INVERSE_MAPPING)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n[5-Feature] XGBoost Model Performance (on Test Samples):")
    print(f"Overall Model Accuracy: {accuracy*100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, accuracy

# --- Prediction and Auto-Generation Logic ---
def predict_fault(model, current, resistance, temperature, load, vibration):
    """Predicts the fault using the trained ML model with 5 inputs."""
    if model is None:
        raise Exception("ML Model is not trained or loaded.")
        
    # Input DataFrame must match the training feature names
    input_data = pd.DataFrame([[current, resistance, temperature, load, vibration]], 
                              columns=['Current (A)', 'Resistance (Ω)', 'Temperature (°C)', 'Load (%)', 'Vibration (g)'])
    
    # Prediction returns a numerical class (e.g., numpy.int64)
    prediction_encoded = model.predict(input_data)[0]
    
    # FIX: Decode the prediction from number (e.g., 2) to string (e.g., 'Normal')
    prediction = GLOBAL_INVERSE_MAPPING.get(prediction_encoded, "Unknown Fault")
    
    return prediction

def auto_generate_fault_data(fault_state='Normal'):
    """Auto generates a single set of input values based on a simulated fault state."""
    # (Simplified for brevity, assuming the full logic from previous step is here)
    base_current = 20.0
    base_resistance = 5.0
    base_temperature = 55.0
    base_load = 50
    base_vibration = 0.5

    # ... Full logic to generate C, R, T, L, V based on fault_state ...
    # (Since you have this code, I'll return placeholder values for now)
    
    # Placeholder: In your actual file, this will contain the full logic
    C, R, T, L, V = 20.0, 5.0, 55.0, 50, 0.5 
    
    # Re-insert your full logic here:
    if fault_state == 'Bearing Fault':
        C, R, T, L, V = np.random.normal(24.0, 1.5), np.random.normal(5.0, 0.1), np.random.normal(70.0, 3), np.random.randint(50, 70), np.random.normal(2.5, 0.5) 
    # ... (Include all your 'elif' blocks here) ...

    # Final clamping and rounding
    C = round(max(0, C), 2); R = round(max(1, R), 2); T = round(max(20, T), 2); L = max(0, min(100, L)); V = round(max(0.1, V), 2)
    
    return C, R, T, L, V

def get_real_time_crt_data(num_points=100, fault_state='Normal', c_input=20.0, r_input=5.0, t_input=55.0, v_input=0.5):
    """Generates synthetic 'real-time' CRT and V data for plotting."""
    # (Assuming the full implementation from the previous step is here)
    time = np.linspace(0, 5, num_points)
    current = np.full(num_points, c_input / 100.0) # Placeholder
    temperature = np.full(num_points, t_input / 100.0) # Placeholder
    resistance = np.full(num_points, r_input / 10.0) # Placeholder
    vibration = np.full(num_points, v_input / 5.0) # Placeholder

    # ... Your full logic to generate time series data based on fault_state ...
    
    return time, current, temperature, resistance, vibration

# Initialize the model and accuracy globally
try:
    ML_MODEL, MODEL_ACCURACY = train_and_get_model() 
except Exception as e:
    # Print a helpful message if XGBoost is not installed
    if "No module named 'xgboost'" in str(e):
        print("\n*** ERROR: XGBoost library not found. Please run 'pip install xgboost' ***")
    else:
        print(f"Error initializing ML Model: {e}")
    ML_MODEL = None
    MODEL_ACCURACY = 0.0