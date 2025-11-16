# ASL Sign Language Recognition Web App

## Project Overview  
This project is a **Sign Language Recognition system** using a **Convolutional Neural Network (CNN)** trained on the ASL Alphabet dataset. It provides a **Streamlit-based web app** for predicting hand signs from images or camera snapshots. Predictions are allowed only **between 6:00 PM – 10:00 PM** (Asia/Kolkata). Users can build words from predicted letters and manage a list of known words.

## Features  
- **CNN-based ASL sign recognition**  
  - Input image size: 64x64 RGB  
  - 29 classes: `A-Z`, `nothing`, `space`, `delete`  
  - Softmax output with top-5 probabilities  

- **Streamlit Web Interface**  
  - Upload images (`jpg`, `jpeg`, `png`)  
  - Take snapshots via webcam  
  - Append predictions to build words  
  - Check and save known words (session-only)  

- **Time restriction for predictions**  
  - Active only from 6:00 PM – 10:00 PM (Asia/Kolkata)  
  - Displays current time and status  

- **Interactive UI**  
  - Tabs for image upload & camera snapshot  
  - Sidebar for model settings and known words  
  - Word builder section with append/clear/reset buttons  
  - Displays last single prediction   

## Project Structure  
- asl_cnn_model.h5  → Trained CNN model
- asl_app.ipynb  → Model training script
- app_gui.py  → Streamlit app for prediction
- readme.md  → Project documentation
- asl_alphabet_train/  → Training dataset (folders A-Z, nothing, space, delete)
- asl_alphabet_test/  → Test dataset (individual images)


##  Installation & Usage  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/Nayana739/AQI-PROJECT.git
   cd AQI-PROJECT
   ```  
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```  
3. **Run the Streamlit app**  
   ```bash
   streamlit run streamlit_app.py
   ```  
4. **Use the API**
- Use the app
- Upload an image or take a camera snapshot
- Predict the ASL sign (enabled only between 6 PM – 10 PM)
- Append predicted letters to the current word
- Check or save words in the known words list 

## Model & Optimization
This model is a **baseline Random Forest Regressor**. Improvements can include:
- Testing **other regression models** (XGBoost, Gradient Boosting, etc.)
- **Hyperparameter tuning** for better accuracy
- Incorporating **more pollutant features or meteorological data**



