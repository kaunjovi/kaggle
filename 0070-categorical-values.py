# https://www.kaggle.com/code/alexisbcook/categorical-variables

import pandas as pd 
from sklearn.model_selection import train_test_split

if __name__ == "__main__" : 
    print("Categorical values")

    # Load the data. There are no index in the file 
    home_data = pd.read_csv("data/intermediate-ml/melb_data.csv") 
    y = home_data.Price
    X = home_data.drop( ["Price"], axis=1 )

    # print(f"{ X.dtypes }")

    # Create training and validation data set. 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, train_size = 0.8, random_state = 0 )
    # X_train, X_val, y_train, y_val = train_test_split(X, y)

    cols_with_missing_values = [col for col in X.columns if X[col].isnull().any()]
    print(f"Columns with missing values { cols_with_missing_values }")
    # Dropping for now. But will trying imputing later. 


    cols_with_low_cardinality = [col for col in X.columns if X[col].nunique() < 10 and X[col].dtype == "object"]
    print(f"Columns of object type with low cardinality { cols_with_low_cardinality }")
    # print(f"{X[cols_with_low_cardinality].head()}")
    
    cols_numeric = [col for col in X.columns if X[col].dtype in ["int64", "float64"]]
    print(f"Columns of numeric type { cols_numeric }")

    # Next step : follow through this tutorial 
