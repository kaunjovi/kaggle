# https://www.kaggle.com/code/dansbecker/random-forests

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__" : 

    # Load data
    iowa_file_path = "data/home-data-for-ml-course/train.csv"
    home_data = pd.read_csv(iowa_file_path)

    # predictions and features. 
    y = home_data.SalePrice
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[features]

    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Attempt 1 : mae = 29,652 -- good 
    iowa_model = DecisionTreeRegressor(random_state=1 )
    iowa_model.fit(train_X, train_y)
    predict = iowa_model.predict(val_X)
    mae = mean_absolute_error( predict, val_y)
    print(f"Attempt#1 - MAE [{ mae }]")

    # Attepmpt 2 : mae = 27,282 -- better 
    iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1 )
    iowa_model.fit(train_X, train_y)
    predict = iowa_model.predict(val_X)
    mae = mean_absolute_error( predict, val_y)
    print(f"Attempt#2 - MAE [{ mae }]")

    # Attempt 3 : mae = 21,857 -- best (or is there better. there better be.)
    iowa_model = RandomForestRegressor(random_state=1 )
    iowa_model.fit(train_X, train_y)
    predict = iowa_model.predict(val_X)
    mae = mean_absolute_error( predict, val_y)
    print(f"Attempt#3 - MAE [{ mae }]")



# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error

# forest_model = RandomForestRegressor(random_state=1)
# forest_model.fit(train_X, train_y)
# melb_preds = forest_model.predict(val_X)
# print(mean_absolute_error(val_y, melb_preds))