# Seismic Data Analysis for Mars and Apollo Lunar Missions

## Overview

This project allows you to process seismic data from both the Mars (InSight mission) and the Apollo lunar missions. The system is designed to handle **Mars data** in **MiniSEED** and **STATIONXML** formats, as well as **Apollo lunar seismic data** in **CSV** format. The application allows users to:
- Upload seismic data from either Mars or Apollo Lunar missions.
- Apply preprocessing techniques like **response correction**, **bandpass filtering**, and **detrending**.
- Use **anomaly detection** techniques for Mars data.
- Choose between **Machine Learning (Random Forest)** and **Deep Learning (1D CNN)** models for analyzing Apollo lunar data.

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```

2. Navigate to the project directory:
    ```bash
    cd seismic_project
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1. Start the **Streamlit** application:
    ```bash
    streamlit run app.py
    ```

2. Open your browser and go to the local Streamlit server (usually `http://localhost:8501/`).

## Usage

1. Select **Mars** or **Apollo Lunar** data.
2. Upload the corresponding files:
   - For **Mars data**, upload a MiniSEED file and an XML metadata file.
   - For **Lunar data**, upload a CSV file.
3. For Lunar data, choose between **Machine Learning** or **Deep Learning** models.
4. View results such as **anomaly detection** for Mars or **model accuracy** for Lunar data.

## File Structure

