import pandas as pd 



if __name__ == "__main__" : 
    # https://www.kaggle.com/code/kaunjovi/exercise-underfitting-and-overfitting/edit 
    print(f"Hello world. Pandas version [{pd.__version__}]")

    # Path of the file to read
    iowa_file_path = './data/home-data-for-ml-course/train.csv'
    home_data = pd.read_csv(iowa_file_path)
    print(f"home_date.shape [{ home_data.shape }]")
