import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


def investigate_data (data, name = "data" ) : 
    print(f"##############")
    print(f"## Investigating data from {name}.")

    print(f"Shape [{ data.shape}]")

    # print (f"## Data Types")
    # print(f"{ data.dtypes}")

    print (f"## Head")
    print(f"{ data.head()}")

    # print (f"## Describe")
    # print(f"{ data.describe()}")

# Function for comparing different models
def score_model(model, X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    return mean_absolute_error(y_valid, predictions)


if __name__ == "__main__" : 
    print(f"Intermediate ML.")
    pd.set_option('display.max_rows', None, 'display.max_columns', None)

    # Load training data 
    train_file_path = "data/intermediate-ml/train.csv"
    train_data = pd.read_csv(train_file_path, index_col= "Id")
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    y = train_data.SalePrice 
    X = train_data[features].copy()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # Define the models
    model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
    model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
    model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
    model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
    model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
    models = [model_1, model_2, model_3, model_4, model_5]

    for model in models : 
        mae = score_model(model, X_train, X_valid, y_train,  y_valid) 
        print(f"MAE [{mae}] Model [{model}]")


    my_model =  RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)# Your code here

    # Fit the model to the training data
    my_model.fit(X, y)

    # Generate test predictions
    test_file_path = "data/intermediate-ml/test.csv"
    test_data = pd.read_csv(test_file_path, index_col="Id")
    X_test = test_data[features].copy()
    predictions_test = my_model.predict(X_test)

    # Save predictions in format used for competition scoring
    output = pd.DataFrame({'Id': X_test.index,
                        'SalePrice': predictions_test})
    output.to_csv('submission.csv', index=False)