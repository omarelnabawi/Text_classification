# Text Classifier Project

This project implements a text classifier using Deep learning techniques and provides a Streamlit web application for easy interaction with the model.

## Project Structure

- `Text_classifier.ipynb`: Jupyter notebook containing the model creation and training process.
- `app.py`: Streamlit web application for interacting with the trained model.
- `model.rar`: Compressed file containing the trained model.
- `test.txt`: Test data for evaluating the model.
- `tokenizer.pkl`: Pickled tokenizer for preprocessing input text.
- `train.txt`: Training data used to train the model.
- `val.txt`: Validation data for model evaluation.
-  `requirements.txt: Requirements needed for the project.
## Setup and Installation

1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Extract the `model.rar` file to obtain the trained model.

## Usage

### Training the Model

To train or modify the model:

1. Open `Text_classifier.ipynb` in Jupyter Notebook or JupyterLab.
2. Follow the instructions in the notebook to train the model.

### Running the Streamlit App

To run the Streamlit web application:

1. Ensure all dependencies are installed and the model is extracted.
2. Run the following command in your terminal:
   ```
   streamlit run app.py
   ```
3. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

## Model Details

The text classifier is trained on the provided training data (`train.txt`) and validated using the validation set (`val.txt`). The model's performance can be evaluated using the test set (`test.txt`).

## Contributing

Feel free to fork this repository and submit pull requests to contribute to this project. For major changes, please open an issue first to discuss what you would like to change.
