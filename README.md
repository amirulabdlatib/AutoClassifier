This project is a Streamlit web application designed for Automated Machine Learning (AutoML) tasks. The application integrates various Python libraries such as Streamlit, Pandas, ydata_profiling, and PyCaret to provide a user-friendly interface for data analysis and machine learning modeling. The key features of the application include:

1. **Data Upload:**
   - Users can upload CSV files through the application.
   - The uploaded data is displayed in a DataFrame for further exploration.

2. **Exploratory Data Analysis (EDA):**
   - The application utilizes the `ydata_profiling` library to generate an automatic EDA report.
   - Insights from the profiling report are displayed, aiding users in understanding the characteristics of their dataset.

3. **Machine Learning Modeling:**
   - The application supports the training of machine learning models using PyCaret.
   - Users can select a target variable and initiate the automated ML pipeline.
   - The best-performing model is saved as 'best_model.pkl'.
   - A download button allows users to retrieve the trained model for future use.

## Dependencies:

The project relies on the following Python libraries:

- Streamlit
- Pandas
- ydata_profiling
- streamlit_pandas_profiling
- pycaret

## How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/amirulabdlatib/AutoClassifier.git
   cd AutoClassifier
   ```

2. **Make a virtual environment**
   ```bash
   pip install virtualenv
   ```

3. **Create your virtualenv**
   ```bash
   virtualenv yourenv
   ```
    
4. **Activate your environment**
   ```bash
   yourenv\scripts\activate
   ```

5. **Install Required Libraries:**
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the Application:**
   ```bash
   streamlit run main.py
   ```
   Replace `your_script_name.py` with the script containing the provided code.

7. **Access the Application:**
   Open your web browser and go to the provided link (usually http://localhost:8501) to access the AutoML application.

Explore and analyze your datasets effortlessly, and leverage automated machine learning capabilities for model training.
