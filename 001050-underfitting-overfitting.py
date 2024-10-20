import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


if __name__ == "__main__" : 
    # https://www.kaggle.com/code/kaunjovi/exercise-underfitting-and-overfitting/edit 
    print(f"Hello world. Pandas version [{pd.__version__}]")

    iowa_file_path = './data/home-data-for-ml-course/train.csv'
    home_data = pd.read_csv(iowa_file_path)
    print(f"home_date.shape [{ home_data.shape }]")

    # The target that we want to predict.
    y = home_data.SalePrice 
    print(f"The price that we would like to be able to predict")
    print(f"{ y.head() }")

    # The features. 
    print(f"The features that we feel should allow us to predict our Sales price")
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[features]
    print(f"{ X.head() }")

    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Specify Model
    iowa_model = DecisionTreeRegressor(random_state=1)
    # Fit Model
    iowa_model.fit(train_X, train_y)

    # Make validation predictions and calculate mean absolute error
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE: {:,.0f}".format(val_mae))



