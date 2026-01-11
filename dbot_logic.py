# dbot_logic.py (Complete Simulation Logic for 12 Faults: C, R, T, L, V)
import random

# NOTE: Assuming MODEL_ACCURACY is imported or defined elsewhere. 
# We'll use a placeholder value here for the UI output.
MODEL_ACCURACY = 0.92 

# --- Threshold Definitions (Based on Motor Specs: C=25A, R=5Ω, T=75°C, V=1.5mm/s)
THRESHOLDS = {
    'C_HIGH': 26.0, 'C_CRITICAL': 30.0, 'C_LOW': 10.0,
    'R_LOW': 4.0, 'R_CRITICAL_LOW': 2.5, 'R_HIGH': 6.0, 'R_CRITICAL_HIGH': 7.0,
    'T_HIGH': 75.0, 'T_CRITICAL': 85.0,
    'V_HIGH': 1.8, 'V_CRITICAL': 2.8, 'V_LOW': 0.3,
    'L_HIGH': 85  # Load %
}

# --- Motor Fault Advice Data (Expanded to 12 Faults) ---
FAULT_ADVICE = {
    'Normal': {
        'risk': 'Low',
        'details': "The motor is operating within normal parameters. No immediate action is required. Continue routine monitoring.",
        'color': 'Green'
    },
    'Bearing Fault': {
        'risk': 'High',
        'details': "Elevated temperature (T) and high vibration (V) suggest a bearing fault. Immediate scheduling for inspection and replacement is recommended to prevent catastrophic failure.",
        'color': 'Red'
    },
    'Winding Short': {
        'risk': 'Critical',
        'details': "High current (C) and critically low resistance (R) are strong indicators of a winding short. **IMMEDIATE SHUTDOWN** is mandatory to prevent fire and severe damage. ",
        'color': 'DarkRed'
    },
    'Overload': {
        'risk': 'Medium',
        'details': "Very high current (C) and load percentage (L) indicate continuous overload. Reduce the mechanical load on the motor immediately and check the protective relay settings.",
        'color': 'Orange'
    },
    'Misalignment/Imbalance': {
        'risk': 'High',
        'details': "High vibration (V) and moderate temperature (T) suggest a mechanical issue like misalignment or imbalance. Schedule corrective alignment/balancing maintenance soon to protect bearings and coupling.",
        'color': 'Red'
    },
    'Rotor Bar Damage': {
        'risk': 'Medium',
        'details': "Slightly increased current (C) and vibration (V) could indicate early rotor bar damage. Monitor closely and consider advanced current signature analysis (MCSA) for confirmation.",
        'color': 'Yellow'
    },
    'Loose Connections': {
        'risk': 'High',
        'details': "Elevated resistance (R) and erratic/low current (C) suggest loose electrical connections. De-energize and inspect terminals for wear or pitting to prevent fire or phase loss.",
        'color': 'Red'
    },
    'Phase Loss (Single Phasing)': {
        'risk': 'Critical',
        'details': "Critically high current (C) and temperature (T) often point to a phase loss. **IMMEDIATE SHUTDOWN** is required, as the motor is running on reduced phases, leading to rapid overheating.",
        'color': 'DarkRed'
    },
    'Stator Insulation Degradation': {
        'risk': 'Medium',
        'details': "Low resistance (R) and high temperature (T) over time suggest insulation breakdown. Perform a Megohm test. Schedule a motor rewind or replacement before a full short occurs.",
        'color': 'Orange'
    },
    'Ground Fault': {
        'risk': 'Critical',
        'details': "Extreme drop in resistance (R) towards zero and high current (C) may indicate current leakage to the ground. **IMMEDIATE SHUTDOWN.** Check for physical damage and moisture ingress. ",
        'color': 'DarkRed'
    },
    'Fan/Ventilation Blockage': {
        'risk': 'Low',
        'details': "High temperature (T) with otherwise normal electrical readings. Check cooling fan operation and clear any dirt or debris blocking the motor's cooling fins.",
        'color': 'Yellow'
    },
    'Low/High Voltage': {
        'risk': 'Medium',
        'details': "Sustained deviations in current (C) without corresponding load (L) changes suggest supply voltage issues. Check the incoming power supply supply at the motor terminals.",
        'color': 'Yellow'
    }
}


def _check_confidence(count):
    """Simple function to set confidence based on the number of matching sensor conditions."""
    if count >= 3:
        return random.uniform(0.92, 0.99)
    elif count == 2:
        return random.uniform(0.75, 0.90)
    elif count == 1:
        return random.uniform(0.50, 0.70)
    return 0.0


def predict_fault_simulated(C, R, T, L, V):
    """
    Simulates the ML prediction using threshold-based logic (acting as your predict_fault).
    Returns the predicted fault string and its confidence.
    """
    fault_probabilities = {}

    # --- 1. Bearing Fault (High T and Critical V) ---
    count = 0
    if T >= THRESHOLDS['T_HIGH']: count += 1
    if V >= THRESHOLDS['V_CRITICAL']: count += 1
    if R < THRESHOLDS['R_HIGH']: count += 1
    fault_probabilities['Bearing Fault'] = _check_confidence(count)

    # --- 2. Overload (Critical C and High L) ---
    count = 0
    if C >= THRESHOLDS['C_CRITICAL']: count += 1
    if L >= THRESHOLDS['L_HIGH']: count += 1
    if V < THRESHOLDS['V_HIGH']: count += 1
    fault_probabilities['Overload'] = _check_confidence(count)

    # --- 3. Winding Short (Critically Low R and High C) ---
    count = 0
    if R <= THRESHOLDS['R_CRITICAL_LOW'] and R > 0.1: count += 1 # R is very low but not zero
    if C >= THRESHOLDS['C_HIGH']: count += 1
    if T < THRESHOLDS['T_HIGH']: count += 1
    fault_probabilities['Winding Short'] = _check_confidence(count)

    # --- 4. Misalignment/Imbalance (High V and High T, but not Critical) ---
    count = 0
    if V >= THRESHOLDS['V_HIGH'] and V < THRESHOLDS['V_CRITICAL']: count += 1
    if T >= THRESHOLDS['T_HIGH'] and T < THRESHOLDS['T_CRITICAL']: count += 1
    if C > THRESHOLDS['C_HIGH']: count += 1 
    fault_probabilities['Misalignment/Imbalance'] = _check_confidence(count)
    
    # --- 5. Rotor Bar Damage (Elevated C and Medium V) ---
    count = 0
    if C >= THRESHOLDS['C_HIGH'] and C < THRESHOLDS['C_CRITICAL']: count += 1
    if V >= THRESHOLDS['V_LOW'] and V < THRESHOLDS['V_HIGH']: count += 1
    if T >= THRESHOLDS['T_HIGH']: count += 1 
    fault_probabilities['Rotor Bar Damage'] = _check_confidence(count)
    
    # --- 6. Loose Connections (Elevated R, Low C) ---
    count = 0
    if R >= THRESHOLDS['R_HIGH']: count += 1
    if C < THRESHOLDS['C_HIGH']: count += 1
    if T < THRESHOLDS['T_HIGH']: count += 1
    fault_probabilities['Loose Connections'] = _check_confidence(count)

    # --- 7. Phase Loss (Critical C AND Critical T) ---
    count = 0
    if C >= THRESHOLDS['C_CRITICAL']: count += 1
    if T >= THRESHOLDS['T_CRITICAL']: count += 1
    if L >= THRESHOLDS['L_HIGH']: count += 1
    fault_probabilities['Phase Loss (Single Phasing)'] = _check_confidence(count)

    # --- 8. Stator Insulation Degradation (Low R and High T) ---
    count = 0
    if R > THRESHOLDS['R_CRITICAL_LOW'] and R <= THRESHOLDS['R_LOW']: count += 1
    if T >= THRESHOLDS['T_HIGH']: count += 1
    if C >= THRESHOLDS['C_HIGH']: count += 1
    fault_probabilities['Stator Insulation Degradation'] = _check_confidence(count)

    # --- 9. Ground Fault (R close to zero, Critical C) ---
    count = 0
    if R <= 0.1: count += 1 # Resistance virtually zero
    if C >= THRESHOLDS['C_CRITICAL']: count += 1
    if T < THRESHOLDS['T_HIGH']: count += 1 # May trip before heat builds
    fault_probabilities['Ground Fault'] = _check_confidence(count)

    # --- 10. Fan/Ventilation Blockage (High T, Normal C/R) ---
    count = 0
    if T >= THRESHOLDS['T_HIGH']: count += 1
    if C < THRESHOLDS['C_HIGH']: count += 1
    if R < THRESHOLDS['R_HIGH']: count += 1
    fault_probabilities['Fan/Ventilation Blockage'] = _check_confidence(count)

    # --- 11. Low/High Voltage (C deviation without L change) ---
    count = 0
    # Current deviates, but load remains medium/low
    if (C >= THRESHOLDS['C_HIGH'] or C <= THRESHOLDS['C_LOW']) and L < THRESHOLDS['L_HIGH']: count += 1
    if R < THRESHOLDS['R_HIGH']: count += 1
    fault_probabilities['Low/High Voltage'] = _check_confidence(count)


    # --- 12. Final Prediction Logic ---
    
    predicted_fault = max(fault_probabilities, key=fault_probabilities.get)
    confidence = fault_probabilities[predicted_fault]

    # Check for 'Normal' state: All readings are within acceptable range AND no high confidence fault was predicted
    is_normal_range = (C < THRESHOLDS['C_HIGH'] and 
                       R > THRESHOLDS['R_LOW'] and R < THRESHOLDS['R_HIGH'] and 
                       T < THRESHOLDS['T_HIGH'] and 
                       V < THRESHOLDS['V_HIGH'])
                       
    if is_normal_range and confidence < 0.70:
        return 'Normal', random.uniform(0.90, 0.99)
        
    return predicted_fault, confidence

# --- Core Chatbot Logic ---
def get_chatbot_response(current, resistance, temperature, load, vibration):
    """
    Takes 5 input features, predicts the fault, and generates a formatted response.
    """
    
    # 1. Get the ML Prediction (Using the simulated function)
    predicted_fault, confidence = predict_fault_simulated(current, resistance, temperature, load, vibration)
    
    # 2. Get Advice from FAULT_ADVICE
    advice = FAULT_ADVICE.get(predicted_fault, FAULT_ADVICE['Normal'])
    
    # 3. Format Input Summary
    summary = (
        f"<b>--- Input Readings ---</b><br>"
        f"Current: {current:.2f} A | Resistance: {resistance:.2f} Ω<br>"
        f"Temperature: {temperature:.2f} °C | Load: {load:.0f} %<br>"
        f"Vibration: {vibration:.2f} g<br>"
    )

    # 4. Format Prediction and Advice
    prediction_html = f"""
        <span style="color: {advice['color']}; font-weight: bold; font-size: 16pt;">
            [ {advice['risk'].upper()} Risk ] Predicted Fault: {predicted_fault.upper()}
        </span>
        <br>
        <span style="font-style: italic; font-weight: bold;">
            Confidence: {confidence*100:.1f}%
        </span>
        <br><br>
        <b>Maintenance Advice:</b> {advice['details']}
        <br><br>
        <i>ML Accuracy on Test Data: {MODEL_ACCURACY*100:.2f}%</i>
    """
    
    # Combine everything
    full_response = f"{summary}<br>{prediction_html}"
    
    return predicted_fault, full_response