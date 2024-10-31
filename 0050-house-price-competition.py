import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


def run_competition_code () : 
    # Load the data, and separate the target
    iowa_file_path = 'data/house-price-kaggle-competition/train.csv'
    home_data = pd.read_csv(iowa_file_path)
    y = home_data.SalePrice

    # Create X (After completing the exercise, you can return to modify this line!)
    # All the int64 features. 
    features = ["MSSubClass" , "LotArea" , "OverallQual" , "OverallCond" , "YearBuilt" , "YearRemodAdd" , "BsmtFinSF1" , "BsmtFinSF2" , "BsmtUnfSF" , "TotalBsmtSF" , "1stFlrSF" , "2ndFlrSF" , "LowQualFinSF" , "GrLivArea" , "BsmtFullBath" , "BsmtHalfBath" , "FullBath" , "HalfBath" , "BedroomAbvGr" , "KitchenAbvGr" , "TotRmsAbvGrd" , "GarageCars" , "GarageArea" , "WoodDeckSF" , "OpenPorchSF" , "EnclosedPorch" , "3SsnPorch" , "ScreenPorch" , "PoolArea" , "MiscVal" , "MoSold" , "YrSold" , "Fireplaces" ]
    # All int64 and float64
    features = ["MSSubClass" , "LotArea" , "OverallQual" , "OverallCond" , "YearBuilt" , "YearRemodAdd" , "BsmtFinSF1" , "BsmtFinSF2" , "BsmtUnfSF" , "TotalBsmtSF" , "1stFlrSF" , "2ndFlrSF" , "LowQualFinSF" , "GrLivArea" , "BsmtFullBath" , "BsmtHalfBath" , "FullBath" , "HalfBath" , "BedroomAbvGr" , "KitchenAbvGr" , "TotRmsAbvGrd" , "GarageCars" , "GarageArea" , "WoodDeckSF" , "OpenPorchSF" , "EnclosedPorch" , "3SsnPorch" , "ScreenPorch" , "PoolArea" , "MiscVal" , "MoSold" , "YrSold" , "Fireplaces", "LotFrontage", "MasVnrArea", "GarageYrBlt" ]

    # Feature set #1
    # features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

    # Select columns corresponding to features, and preview the data
    X = home_data[features]
    X.head()

    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Define a random forest model
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(train_X, train_y)
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

    rf_model_on_full_data = rf_model.fit(X, y)

    # path to file you will use for predictions
    test_data_path = 'data/house-price-kaggle-competition/test.csv'

    # read test data file using pandas
    test_data = pd.read_csv(test_data_path)
    # pd.set_option('display.max_rows', None, 'display.max_columns', None)

    # test_data.dtypes

    # # create test_X which comes from test_data but includes only the columns you used for prediction.
    # # The list of columns is stored in a variable called features
    test_X = test_data[features]

    # # make predictions which we will submit. 
    test_preds = rf_model_on_full_data.predict(test_X)


    # Run the code to save predictions in the format used for competition scoring
    output = pd.DataFrame({'Id': test_data.Id,
                        'SalePrice': test_preds})
    output.to_csv('submission.csv', index=False)

if __name__ == "__main__" : 
    # run_competition_code() 

    pd.set_option('display.max_rows', None, 'display.max_columns', None)

    # Load training data. 
    iowa_file_path = 'data/house-price-kaggle-competition/train.csv'
    home_data = pd.read_csv(iowa_file_path)
    # print(f"{home_data.describe()}")
    # home_data = home_data.dropna(axis= 0 )
    # print(f"{home_data.describe()}")
    
    # # No surprises about predict 
    y = home_data.SalePrice

    
    # # find all the columns with numerical values 
    # # print(f"{home_data.describe()}")
    # print(f"{ home_data.dtypes}")

    # features = ["MSSubClass" , "LotArea" , "OverallQual" , "OverallCond" , "YearBuilt" , "YearRemodAdd" , "BsmtFinSF1" , "BsmtFinSF2" , "BsmtUnfSF" , "TotalBsmtSF" , "1stFlrSF" , "2ndFlrSF" , "LowQualFinSF" , "GrLivArea" , "BsmtFullBath" , "BsmtHalfBath" , "FullBath" , "HalfBath" , "BedroomAbvGr" , "KitchenAbvGr" , "TotRmsAbvGrd" , "GarageCars" , "GarageArea" , "WoodDeckSF" , "OpenPorchSF" , "EnclosedPorch" , "3SsnPorch" , "ScreenPorch" , "PoolArea" , "MiscVal" , "MoSold" , "YrSold" , "Fireplaces" ]

    # # Select columns corresponding to features, and preview the data
    # X = home_data[features]
    # X.head()

    # # Split into validation and training data
    # train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # # Define a random forest model
    # rf_model = RandomForestRegressor(random_state=1)
    # rf_model.fit(train_X, train_y)
    # rf_val_predictions = rf_model.predict(val_X)
    # rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

    # print(f"MAE [{rf_val_mae}]")

