# https://www.kaggle.com/code/kaunjovikaggle/exercise-model-validation/edit

# use train_test_split to get the correct view of MAE
# if you use in sample, MAE might be 0, too good to be true. 

import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split

def investigate_data (data, name = "data" ) : 
    print(f"##############")
    print(f"## Investigating data from {name}.")

    print (f"## Data Types")
    print(f"{ data.dtypes}")

    print (f"## Head")
    print(f"{ data.head()}")

    print (f"## Describe")
    print(f"{ data.describe()}")


def check_MAE_for_in_sample_predictions(home_data) : 
    # Features and prediction
    y = home_data.SalePrice 
    feature_names = ["LotArea" , "YearBuilt" , "1stFlrSF" , "2ndFlrSF" , "FullBath" , "BedroomAbvGr" , "TotRmsAbvGrd"]
    X = home_data[feature_names]

    # Bring out the big guns 
    iowa_model = DecisionTreeRegressor(random_state=1)
    iowa_model.fit(X,y)
    
    predictions_in_sample = iowa_model.predict(X.head()) 
    print(f"In sample predictions [{ predictions_in_sample}]")

    # [208500. 181500. 223500. 140000. 250000.]
    # [208500. 181500. 223500. 140000. 250000.]

    print(f"Actual target value [{ y.head().to_list() }]")

    # compare actual to predictions. 
    # since it is in sample, mean absolute error will be too low 
    # Mean absolute error 0.0
    print(f"Mean absolute error { mean_absolute_error(y.head(), predictions_in_sample)}")

def check_MAE_for_out_sample_predictions(home_data) : 
    # Features and prediction
    y = home_data.SalePrice 
    feature_names = ["LotArea" , "YearBuilt" , "1stFlrSF" , "2ndFlrSF" , "FullBath" , "BedroomAbvGr" , "TotRmsAbvGrd"]
    X = home_data[feature_names]

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state= 1)

    # Bring out the big guns 
    iowa_model = DecisionTreeRegressor(random_state=1)
    iowa_model.fit(train_X, train_y)
    
    predictions = iowa_model.predict(val_X) 

    print(f"Out of sample predictions [{ predictions}]")
    print(f"Actual target value [{ val_y.head() }]")


    # compare actual to predictions. 
    print(f"Mean absolute error { mean_absolute_error(val_y, predictions)}")


if __name__ == "__main__" : 

    pd.set_option('display.max_rows', None, 'display.max_columns', None)

    # Load data 
    iowa_file_path = "data/home-data-for-ml-course/train.csv"
    home_data = pd.read_csv(iowa_file_path)

    check_MAE_for_in_sample_predictions(home_data)

    check_MAE_for_out_sample_predictions(home_data)
