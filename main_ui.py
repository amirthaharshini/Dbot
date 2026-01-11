# main_ui.py (PyQt6 - Final Version with 5 Features, Corrected Imports, and Visualization Logic)
import sys
import pyqtgraph as pg
import random

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QTextEdit, QFrame, QCheckBox, 
    QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor

# Import modules from Dbot files
from dbot_logic import get_chatbot_response
from dbot_data import get_real_time_crt_data, auto_generate_fault_data, N_ROWS, MODEL_ACCURACY

# --- Dashboard Placeholder Class ---
class DashboardWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Dbot Maintenance Dashboard")
        self.setGeometry(200, 200, 700, 400)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        dashboard_label = QLabel("📊 Dbot Historical Data & Status Dashboard")
        dashboard_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        dashboard_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(dashboard_label)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        
        # IMPROVED DASHBOARD PLACEHOLDER
        info_text.setText(
            "### 📜 Historical Log Overview\n\n"
            "This window will display detailed operational logs for the motor, "
            "including a chronological history of sensor readings (C, R, T, L, V), "
            "predicted fault events, and the system's confidence scores.\n\n"
            "Future development will connect this to a permanent database (SQL/NoSQL) "
            "for persistent data analysis and reporting."
        )
        layout.addWidget(info_text)

# --- Main Application Class ---
class DbotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # DEBUG PRINT 1
        print("DEBUG: DbotApp __init__ start.") 
        
        self.setWindowTitle("🤖 Dbot - Smart Motor Troubleshooting")
        self.setGeometry(100, 100, 850, 850)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        
        self.dashboard_window = None 
        self.setup_ui()
        
        # DEBUG PRINT 2
        print("DEBUG: DbotApp __init__ end.")
        
    def setup_ui(self):
        """Sets up all the UI components in the main window."""
        
        # DEBUG PRINT 3
        print("DEBUG: DbotApp setup_ui start.")
        
        # 1. Header Frame
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #F08080; border-radius: 10px;")
        header_layout = QVBoxLayout(header_frame)
        
        title_label = QLabel("🤖 Dbot - Smart Motor Troubleshooting")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title_label)
        self.main_layout.addWidget(header_frame)
        
        # 2. Input and Auto Mode Frame
        input_group_box = QFrame()
        input_group_box.setStyleSheet("border: 1px solid #ccc; border-radius: 8px; padding: 10px;")
        input_layout = QGridLayout(input_group_box)
        
        self.input_fields = {}
        self.auto_mode_checkboxes = {}
        
        # 5 FEATURES
        input_labels = ['Current (A)', 'Resistance (Ω)', 'Temperature (°C)', 'Load (%)', 'Vibration (g)']
        
        for i, label_text in enumerate(input_labels):
            label = QLabel(f"Enter {label_text}:")
            label.setFixedWidth(150)
            input_layout.addWidget(label, i, 0)
            
            entry = QLineEdit()
            
            if label_text == 'Vibration (g)':
                entry.setPlaceholderText("e.g., 0.8")
            else:
                 entry.setPlaceholderText(f"e.g., {'20.5' if 'Current' in label_text else '5.2'}")
                 
            self.input_fields[label_text] = entry
            input_layout.addWidget(entry, i, 1)

            checkbox = QCheckBox("Auto Mode")
            self.auto_mode_checkboxes[label_text] = checkbox
            input_layout.addWidget(checkbox, i, 2)
            
        self.main_layout.addWidget(input_group_box)
        
        # 3. Button Frame
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        
        self.auto_btn = QPushButton("🧪 Auto-Generate Data")
        self.auto_btn.clicked.connect(self.auto_generate_input)
        self.auto_btn.setStyleSheet("background-color: #E7A3D7; color: black; padding: 10px;")
        
        self.diagnose_btn = QPushButton("🧠 Diagnose Fault")
        self.diagnose_btn.clicked.connect(self.diagnose_fault)
        self.diagnose_btn.setStyleSheet("background-color: #F7786B; color: black; padding: 10px;")
        
        self.dashboard_btn = QPushButton("⚙️ Open Dashboard")
        self.dashboard_btn.clicked.connect(self.open_dashboard)
        self.dashboard_btn.setStyleSheet("background-color: #6B8E23; color: white; padding: 10px;")
        
        button_layout.addWidget(self.auto_btn)
        button_layout.addWidget(self.diagnose_btn)
        button_layout.addWidget(self.dashboard_btn)
        self.main_layout.addWidget(button_frame)

        # 4. Chatbot Output
        chatbot_header = QLabel("🔮 Dbot Predictive Maintenance Output")
        chatbot_header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.main_layout.addWidget(chatbot_header)
        
        self.chatbot_text = QTextEdit()
        self.chatbot_text.setReadOnly(True)
        self.chatbot_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.chatbot_text.setFixedHeight(180) 
        self.main_layout.addWidget(self.chatbot_text)
        
        self.initial_welcome_message()

        # 5. Graph Area (using pyqtgraph)
        graph_header = QLabel("📊 Real-Time Monitoring (Simulated)")
        graph_header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.main_layout.addWidget(graph_header)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Normalized Values')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        
        # VISUALIZATION FIX: Initialize graph to a flat STANDBY state
        time_data = [0, 1, 2, 3, 4, 5]
        default_normalized_value = 0.3
        
        self.plot_widget.setYRange(0, 0.7) 
        self.plot_widget.setXRange(min(time_data), max(time_data))
        
        self.curve_c = self.plot_widget.plot(time_data, [default_normalized_value] * 6, pen=pg.mkPen('r', width=2), name='Current (A)')
        self.curve_t = self.plot_widget.plot(time_data, [default_normalized_value] * 6, pen=pg.mkPen('b', width=2), name='Temperature (°C)')
        self.curve_r = self.plot_widget.plot(time_data, [default_normalized_value] * 6, pen=pg.mkPen('g', width=2), name='Resistance (Ω)')
        self.curve_v = self.plot_widget.plot(time_data, [default_normalized_value] * 6, pen=pg.mkPen('m', width=2), name='Vibration (g)')

        self.plot_widget.addLegend()
        self.main_layout.addWidget(self.plot_widget)
        self.plot_widget.setTitle(f'<span style="color: black; font-size: 14pt;">📈 Real-Time Monitoring Graph (Status: STANDBY)</span>')

        # DEBUG PRINT 4
        print("DEBUG: DbotApp setup_ui end.")


    def initial_welcome_message(self):
        """Displays the welcome message including the ML accuracy."""
        accuracy_msg = f"The underlying ML Algorithm achieves an estimated <b>{MODEL_ACCURACY*100:.2f}% accuracy</b> on unseen data (Dataset Size: {N_ROWS})."
        
        welcome_text = (
            f"Welcome to Dbot! {accuracy_msg} Enter values or click '🧪 Auto-Generate' to start a diagnosis."
        )
        self.chatbot_text.setText(welcome_text)
        
    
    def _update_graph_on_diagnosis(self, fault, C, R, T, V): 
        """Updates the graph to reflect the diagnosed fault trend based on input values."""
        
        # This function is ONLY called inside diagnose_fault, meeting the requirement.
        time, current, temperature, resistance, vibration = get_real_time_crt_data(
            fault_state=fault, c_input=C, r_input=R, t_input=T, v_input=V
        )

        # Update the plot curves
        self.curve_c.setData(time, current)
        self.curve_t.setData(time, temperature)
        self.curve_r.setData(time, resistance)
        self.curve_v.setData(time, vibration) 
        
        # Update title color
        color = QColor('red') if fault != 'Normal' else QColor('black')
        title_html = f'<span style="color: {color.name()}; font-size: 14pt;">📈 Real-Time Monitoring Graph (Status: {fault.upper()})</span>'
        self.plot_widget.setTitle(title_html)


    def auto_generate_input(self):
        """Generates a random set of input values and fills the fields."""
        
        fault_states = ['Normal', 'Bearing Fault', 'Overload', 'Winding Short', 'Misalignment/Imbalance', 'Rotor Bar Damage']
        random_state = random.choice(fault_states)
        
        C, R, T, L, V = auto_generate_fault_data(random_state) 
        
        self.input_fields['Current (A)'].setText(f"{C:.2f}")
        self.input_fields['Resistance (Ω)'].setText(f"{R:.2f}")
        self.input_fields['Temperature (°C)'].setText(f"{T:.2f}")
        self.input_fields['Load (%)'].setText(f"{L:.0f}")
        self.input_fields['Vibration (g)'].setText(f"{V:.2f}") 
        
        self.chatbot_text.clear()
        self.chatbot_text.setText(f"Input values have been auto-generated for a '<b>{random_state}</b>' simulation. Click '🧠 Diagnose Fault' to run the ML prediction.")

    
    def diagnose_fault(self):
        """Collects input, runs the diagnosis, and updates the chatbot output and graph."""
        
        try:
            # 1. Get Input and handle Auto-Mode logic
            current_str = self.input_fields['Current (A)'].text().strip()
            resistance_str = self.input_fields['Resistance (Ω)'].text().strip()
            temperature_str = self.input_fields['Temperature (°C)'].text().strip()
            load_str = self.input_fields['Load (%)'].text().strip()
            vibration_str = self.input_fields['Vibration (g)'].text().strip() 

            C_auto, R_auto, T_auto, L_auto, V_auto = auto_generate_fault_data('Normal')
            auto_generated = False
            auto_mode_active = any(cb.isChecked() for cb in self.auto_mode_checkboxes.values())

            # Check if auto mode is active or any field is empty 
            if auto_mode_active or not (current_str and resistance_str and temperature_str and load_str and vibration_str):
                if all(cb.isChecked() for cb in self.auto_mode_checkboxes.values()):
                    random_fault_state = random.choice(['Normal', 'Bearing Fault', 'Overload', 'Winding Short', 'Misalignment/Imbalance', 'Rotor Bar Damage'])
                    C_auto, R_auto, T_auto, L_auto, V_auto = auto_generate_fault_data(random_fault_state) 
                    auto_generated = True
                else: 
                    C_auto, R_auto, T_auto, L_auto, V_auto = auto_generate_fault_data('Normal') 

            # Assign final values
            C = float(current_str) if current_str else C_auto
            R = float(resistance_str) if resistance_str else R_auto
            T = float(temperature_str) if temperature_str else T_auto
            L = float(load_str) if load_str else L_auto
            V = float(vibration_str) if vibration_str else V_auto
            
            if auto_generated:
                self.input_fields['Current (A)'].setText(f"{C:.2f}")
                self.input_fields['Resistance (Ω)'].setText(f"{R:.2f}")
                self.input_fields['Temperature (°C)'].setText(f"{T:.2f}")
                self.input_fields['Load (%)'].setText(f"{L:.0f}")
                self.input_fields['Vibration (g)'].setText(f"{V:.2f}")

            # 2. Prediction 
            fault, response = get_chatbot_response(C, R, T, L, V)
            
            # 3. Update Chatbot Output
            self.chatbot_text.clear()
            self.chatbot_text.setText(response.replace('\n', '<br>') )

            # 4. Update Graph (This is the only place the graph updates, meeting the requirement)
            self._update_graph_on_diagnosis(fault, C, R, T, V)
            
        except ValueError:
            error_msg = "❗ INPUT ERROR: Please ensure all required fields contain valid numerical data (e.g., 20.5, 0.8)."
            self.chatbot_text.clear()
            self.chatbot_text.setText(f'<span style="color: red;">{error_msg}</span>')
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during diagnosis: {e}"
            self.chatbot_text.clear()
            self.chatbot_text.setText(f'<span style="color: red;">An ERROR occurred: {e}</span>')


    def open_dashboard(self):
        """Initializes and shows the placeholder Dashboard window."""
        if self.dashboard_window is None:
            self.dashboard_window = DashboardWindow(self)
        self.dashboard_window.show()
        
if __name__ == "__main__":
    
    # DEBUG PRINT 5
    print("DEBUG: Starting QApplication setup.")
    
    app = QApplication(sys.argv)
    window = DbotApp()
    
    # DEBUG PRINT 6
    print("DEBUG: Showing window and starting exec loop.")
    
    window.show()
    sys.exit(app.exec())