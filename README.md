# EDA App Template

## Getting Started

### Create conda environment
- Name `-n eda`
- Python version = 3.10

> Change the environment name and the version as you please.

```
conda create -n eda python=3.10
```

### Activate the environment
```
conda activate eda
```

### Removing unneeded environment 
```
conda env remove -n env_name

Example
conda env remove -n eda
```

### View required libraries
- All required libraries are listed in the requirements.txt file
- We can view the contents using `wget`like so:
```
wget https://raw.githubusercontent.com/tmbuza/my-app-template/main/requirements.txt
```

> Change the content in the `requirements.txt` to match the libraries installed on your system.

### Install required libraries
```
pip install -r requirements.txt
```

###  Download and unzip the GitHub repo
```
Link to zipped file: https://github.com/tmbuza/my-app-template/archive/refs/heads/main.zip

- Link to zipped file: https://github.com/tmbuza/my-app-template/archive/refs/heads/main.zip
- If you have git installed you can clone the repo and it will unzip automatically to the my-app-template folder.
- Move to the project directory to allow the app to easily access the files.
```
git clone https://github.com/tmbuza/my-app-template.git
cd my-app-template
```

###  Using streamlit to Launch the app
```
streamlit run eda-app.py
```
