import streamlit as st
from io import StringIO
import pandas as pd
from mars_module import detect_anomalies  # Import the Mars anomaly detection function
from lunar_module import preprocess_lunar_data, train_lunar_ml_model, train_lunar_dl_model

# App title
st.title("Lunar/Mars Seismic Data Analysis")

# Dataset selection
dataset_type = st.selectbox("Select Dataset Type", ["Apollo Lunar", "Mars"])

if dataset_type == "Mars":
    # File upload for Mars data (MiniSEED + XML)
    mseed_file = st.file_uploader("Upload your Mars MiniSEED seismic data", type=["mseed"])
    xml_file = st.file_uploader("Upload your Mars station XML metadata", type=["xml"])
    
    if mseed_file is not None and xml_file is not None:
        st.write("Processing Mars seismic data...")

        # Call the function to detect anomalies in Mars seismic data
        try:
            anomalies = detect_anomalies(mseed_file, xml_file)
            st.write(f"Anomalies detected and plotted. Check the generated graph.")
        except Exception as e:
            st.error(f"Error processing Mars seismic data: {e}")

elif dataset_type == "Apollo Lunar":
    # File upload for Apollo Lunar data
    uploaded_file = st.file_uploader("Upload your Apollo Lunar seismic data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        st.write("Processing Apollo Lunar seismic data...")
        lunar_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
        lunar_data = pd.read_csv(lunar_data)
        
        # Pass URN for data simulation if necessary
        urn = "urn:dummy_lunar_data"  # You can modify this to some unique identifier
        filtered_amplitude, time = preprocess_lunar_data(lunar_data, urn=urn)
        
        # Choose model type for Lunar data
        model_type = st.selectbox("Choose a model type", ["Machine Learning (Random Forest)", "Deep Learning (1D CNN)"])
        
        if model_type == "Machine Learning (Random Forest)":
            clf, accuracy = train_lunar_ml_model(filtered_amplitude)
            st.write(f"Lunar (ML) Model Accuracy: {accuracy * 100:.2f}%")
        
        elif model_type == "Deep Learning (1D CNN)":
            model, history, test_acc = train_lunar_dl_model(filtered_amplitude)
            st.write(f"Lunar (DL) Model Test Accuracy: {test_acc * 100:.2f}%")

