# Reference 
# https://www.kaggle.com/code/kaunjovikaggle/exercise-explore-your-data/edit

import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__" : 
    print(f"Pandas version {pd.__version__}")
    # Set display options to show all columns
    pd.set_option('display.max_columns', None) 

    # data_file = "data/home-data-for-ml-course/train.csv"
    # training_data = pd.read_csv(data_file)
    # print(f"{training_data.describe()}")


    data_file = "data/home-data-for-ml-course/melb_data.csv"
    melbourne_data = pd.read_csv(data_file)

    # Show me the columns with data types and some sample data. 
    # print(f"{training_data.columns}") # does not show data type 
    print(f"{melbourne_data.dtypes}")
    print(f"{melbourne_data.head()}")

    # Describe the data
    # print(f"{melbourne_data.describe()}")

    # There are some missing data. Drop the rows with missing values. 
    melbourne_data = melbourne_data.dropna(axis= 0 )

    # Describe again 
    print(f"{melbourne_data.describe()}")

    # Pick the target for prediction 
    y = melbourne_data.Price

    # and your features
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    X = melbourne_data[melbourne_features]

    # Try DecisionTreeRegressor with the same random_state. 
    #For model reproducibility, set a numeric value for random_state when specifying the model
    melbourne_model = DecisionTreeRegressor(random_state=1)
    melbourne_model.fit(X, y)

    print("Making predictions for the following 5 houses:")
    print(X.head())
    prediction = melbourne_model.predict(X.head())
    print("The predictions are")
    print(prediction)

    # How good are the predictions 
    print(f"Mean Absolute Error [{ mean_absolute_error(prediction, y.head())}]")

    

