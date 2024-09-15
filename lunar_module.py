import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Step 1: Simulate seismic data if columns are missing
def simulate_seismic_data(urn):
    """
    Simulate time and amplitude data for seismic waves.
    You can replace this with actual data fetching logic based on the URN.
    """
    np.random.seed(hash(urn) % 123456)  # Seed based on URN to get reproducible results
    time = np.linspace(0, 1000, 1000)  # Simulate time data (1000 points)
    amplitude = np.sin(0.02 * np.pi * time) + np.random.normal(0, 0.5, len(time))  # Simulate seismic wave + noise
    return time, amplitude

# Step 2: Low-pass filter for denoising the seismic data
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Step 3: Preprocess lunar data or simulate data if necessary
def preprocess_lunar_data(data, urn=None):
    """
    Preprocess lunar seismic data or simulate data if 'Time' and 'Amplitude' are not found.
    """
    if 'Time' in data.columns and 'Amplitude' in data.columns:
        time = data['Time'].values
        amplitude = data['Amplitude'].values
    else:
        print("Warning: 'Time' and 'Amplitude' columns not found. Simulating seismic data.")
        time, amplitude = simulate_seismic_data(urn)

    filtered_amplitude = butter_lowpass_filter(amplitude, cutoff=0.1, fs=100, order=4)
    return filtered_amplitude, time

# Step 4: Train ML model (Random Forest)
def train_lunar_ml_model(filtered_amplitude):
    labels = np.random.choice([0, 1], size=len(filtered_amplitude))  # Simulate labels for seismic event detection
    X = filtered_amplitude.reshape(-1, 1)  # Feature matrix
    y = labels  # Labels (0 = no event, 1 = seismic event)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, accuracy

# Step 5: Train DL model (1D CNN)
def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (0 = no event, 1 = seismic event)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_lunar_dl_model(filtered_amplitude):
    labels = np.random.choice([0, 1], size=len(filtered_amplitude))  # Simulate labels
    X = filtered_amplitude.reshape(-1, 1, 1)  # Reshape for CNN input
    y = labels  # Simulated labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_model(input_shape)
    
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    return model, history, test_acc
