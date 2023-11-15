![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)


# gendered-nouns-project

A look into gendered nouns (masculine, feminine, neuter) across a few languages (French, Spanish, German, Polish).
Can a classifier trained on X Y Z accurately predict the gender of nouns from A B C ?

## Goal
Using pre trained models and fine-tuning them on large amounts of data (nouns), to predict genered nouns

## How to Use
### Two Options
### 1. Command Line
Must have `pipenv` installed on machine.

1. Either clone or download the repo
2. `cd` into the folder
3. activate the virtual environment with `pipenv shell`
4. run the following command followed by the approraite flags listed in step 5. `python3 src/main.py <flags go here>`
5. In order to specify which language(s) to train on , which to evaluate on, and which model(s) to use, you must follow this convention for flags:
   - `-t` or `--train` followed by any or all of the following acceptable language codes `fr es pl de` with spaces between them
   - `-e` or `--evaluate` followed by any or all of the following acceptable language codes `fr es pl de` with spaces between them
   - `-m` or `--model` followed by any or all of the following acceptable models  `Bert` (for now)  with spaces between them
  
Example: `python3 src/main.py -t es fr -e fr -m Bert` will train French and Spanish (`-t fr es`) on a fine-tuned Bert model (`-m Bert`) and evaluate on French (`-e fr`)
### 2. Via Streamlit app
(in progress)
