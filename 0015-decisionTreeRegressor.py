# https://www.kaggle.com/code/kaunjovikaggle/exercise-your-first-machine-learning-model/edit

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd 

def investigate_data (data, name = "data" ) : 
    print(f"##############")
    print(f"## Investigating data from {name}.")

    print (f"## Data Types")
    print(f"{ data.dtypes}")

    print (f"## Head")
    print(f"{ data.head()}")

    print (f"## Describe")
    print(f"{ data.describe()}")


if __name__ == "__main__" : 
    # print(f"Hello world from DecisionTreeRegressor") 
    # Set display options to show all rows and column in command prompt. 
    pd.set_option('display.max_rows', None, 'display.max_columns', None)

    # Load data 
    iowa_file_path = "data/home-data-for-ml-course/train.csv"
    home_data = pd.read_csv(iowa_file_path)
    investigate_data(home_data, "Home data")

    # Features and prediction
    y = home_data.SalePrice 
    investigate_data(y, "Sale Price")
    # Create the list of features below
    feature_names = ["LotArea" , "YearBuilt" , "1stFlrSF" , "2ndFlrSF" , "FullBath" , "BedroomAbvGr" , "TotRmsAbvGrd"]

    # Select data corresponding to features in feature_names
    X = home_data[feature_names]
    investigate_data(X, "Features")

    # Bring out the big guns 
    iowa_model = DecisionTreeRegressor(random_state=1)
    iowa_model.fit(X,y)
    predictions = iowa_model.predict(X) 

    # compare actual to predictions. 
    # print(f"Ground truth \n{y.head()} \n{y.tail()}")
    # print(f"Predictions {predictions}")
    print(f"Mean absolute error { mean_absolute_error(y, predictions)}")