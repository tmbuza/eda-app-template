# my-app-template
# A Template for building an EDA App using Python and Streamlit.

## Getting Started

### Create conda environment
- Name `-n eda`
- Python version = 3.10
```
conda create -n eda python=3.10
```

### Activate the environment
```
conda activate eda
```

### View required libraries
- All required libraries are listed in the requirements.txt file
- We can view the contents using `wget`like so:
```
wget https://raw.githubusercontent.com/tmbuza/my-app-template/main/requirements.txt
```

### Install required libraries
```
pip install -r requirements.txt
```

###  Download and unzip the GitHub repo
```
Link to zipped file: https://github.com/tmbuza/my-app-template/archive/refs/heads/main.zip
```

###  Using streamlit to Launch the app
```
streamlit run eda-simple-app.py
```