import obspy
from obspy import read
from lxml import etree
import numpy as np
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Load Seismic Data from MiniSEED File
def load_mars_data(mseed_file, xml_file):
    """
    Load seismic data from MiniSEED and XML files.
    Perform response correction, detrending, and apply a bandpass filter.
    """
    # Load the MiniSEED file
    st = read(mseed_file)
    tr = st[0]

    # Parse XML Metadata for additional context
    tree = etree.parse(xml_file)
    root = tree.getroot()
    metadata = {}
    for element in root.iter("SeisMeta"):
        metadata['station'] = element.find("station").text
        metadata['sensor_type'] = element.find("sensor_type").text
        metadata['start_time'] = element.find("start_time").text
        metadata['end_time'] = element.find("end_time").text
        print(f"Station: {metadata['station']}")
        print(f"Sensor Type: {metadata['sensor_type']}")
        print(f"Start Time: {metadata['start_time']}")
        print(f"End Time: {metadata['end_time']}")

    # Apply Response Correction
    try:
        inv = obspy.read_inventory(xml_file, format="STATIONXML")
        tr.remove_response(inventory=inv, output="VEL", pre_filt=(0.1, 0.2, 20.0, 40.0))
    except Exception as e:
        print(f"Error reading inventory: {e}")

    # Detrend the data to remove linear trends
    tr.detrend(type="linear")

    return tr

# Step 2: Bandpass Filter for Noise Reduction
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sos

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    return sosfiltfilt(sos, data)

# Step 3: Preprocessing (Reshape and Scaling)
def preprocess_data(filtered_data):
    # Reshape Data for Model Input
    data = filtered_data.reshape(-1, 1)

    # Min-Max Scaling (Common in DL Models)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled

# Step 4: Autoencoder for Anomaly Detection
def autoencoder_anomaly_detection(data_scaled, epochs=50):
    input_dim = data_scaled.shape[1]
    encoding_dim = 64

    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train the Autoencoder
    autoencoder.fit(data_scaled, data_scaled, epochs=epochs, batch_size=256, shuffle=True, validation_split=0.1)

    # Reconstruction Error for Anomaly Detection
    reconstructed_data = autoencoder.predict(data_scaled)
    reconstruction_error = np.mean(np.abs(reconstructed_data - data_scaled), axis=1)

    # Anomaly Detection Threshold
    threshold = np.percentile(reconstruction_error, 95)
    anomalies = reconstruction_error > threshold

    return anomalies, threshold

# Step 5: CNN Autoencoder for Anomaly Detection
def cnn_autoencoder_anomaly_detection(data_scaled, epochs=50):
    input_dim = data_scaled.shape[1]

    input_layer = layers.Input(shape=(input_dim, 1))
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(input_layer)
    x = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu')(x)
    encoded = x

    x = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu')(encoded)
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    decoded = layers.Conv1D(1, kernel_size=3, padding='same', activation='sigmoid')(x)

    cnn_autoencoder = models.Model(input_layer, decoded)
    cnn_autoencoder.compile(optimizer='adam', loss='mse')

    # Reshape data for CNN
    data_cnn = data_scaled.reshape(-1, input_dim, 1)

    # Train CNN Autoencoder
    cnn_autoencoder.fit(data_cnn, data_cnn, epochs=epochs, batch_size=256, shuffle=True, validation_split=0.1)

    # Reconstruction Error
    reconstructed_data_cnn = cnn_autoencoder.predict(data_cnn)
    reconstructed_data_cnn = reconstructed_data_cnn.reshape(-1, input_dim)
    reconstruction_error_cnn = np.mean(np.abs(reconstructed_data_cnn - data_scaled), axis=1)

    return reconstruction_error_cnn

# Step 6: Plot Anomalies
def plot_anomalies(time, data, anomalies):
    plt.figure(figsize=(12, 6))
    plt.plot(time, data, 'r', label='Filtered Data')
    plt.scatter(time[anomalies], data[anomalies], color='blue', marker='x', label='Anomalies')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Anomalies Detected in Mars Seismic Data')
    plt.legend()
    plt.show()

# Step 7: Main Function to Process Data and Detect Anomalies
def detect_anomalies(mseed_file, xml_file):
    tr = load_mars_data(mseed_file, xml_file)

    # Bandpass Filter
    lowcut, highcut = 0.1, 20.0
    filtered_data = bandpass_filter(tr.data, lowcut, highcut, tr.stats.sampling_rate)

    # Preprocess the Data
    data_scaled = preprocess_data(filtered_data)

    # Perform Anomaly Detection with Autoencoder
    anomalies_autoencoder, threshold_autoencoder = autoencoder_anomaly_detection(data_scaled)

    # Perform Anomaly Detection with CNN Autoencoder
    reconstruction_error_cnn = cnn_autoencoder_anomaly_detection(data_scaled)

    # Combine Anomalies (from multiple methods)
    combined_anomalies = anomalies_autoencoder | (reconstruction_error_cnn > threshold_autoencoder)

    # Plot the Combined Anomalies
    plot_anomalies(tr.times("matplotlib"), filtered_data, combined_anomalies)

    return combined_anomalies

