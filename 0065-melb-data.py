import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def score_prediction_accuracy(X_train, X_valid, y_train, y_valid) : 
    rt_model = RandomForestRegressor( n_estimators= 10, random_state=0)
    rt_model.fit( X_train, y_train)
    predictions = rt_model.predict(X_valid)
    return mean_absolute_error( predictions, y_valid)


if __name__ == "__main__" : 
    print(f"Panda version [{pd.__version__}]")
    pd.set_option('display.max_rows', None, 'display.max_columns', None)

    # Load data. There is no index coln. 
    melb_data = pd.read_csv("data/intermediate-ml/melb_data.csv" )

    # print(f"Shape = [{ melb_data.shape}]")
    # Shape = [(13580, 21)]

    # print(f"###Columns with datatypes \n{ melb_data.dtypes}")
    # ###Columns with datatypes 
    # Suburb            object
    # Address           object
    # Rooms              int64
    # Type              object
    # Price            float64 <--- Target 
    # Method            object
    # SellerG           object
    # Date              object
    # Distance         float64
    # Postcode         float64
    # Bedroom2         float64
    # Bathroom         float64
    # Car              float64
    # Landsize         float64
    # BuildingArea     float64
    # YearBuilt        float64
    # CouncilArea       object
    # Lattitude        float64
    # Longtitude       float64
    # Regionname        object
    # Propertycount    float64

    # print(f"###Describe \n{ melb_data.describe()}")

    y = melb_data.Price

    X = melb_data.select_dtypes(exclude=["object"]).drop(["Price"],axis=1)

    # print(f"X = {X.head()}")
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, train_size= 0.8, random_state=0)

    # Approach 1 
    # Do nothing about missing values. 
    # MAE = 180484
    print(f"MAE = [{ score_prediction_accuracy(X_train, X_valid, y_train, y_valid)}]")

    # Approach 2 
    # drop the rows with missing values 
    # MAE = 192,451 - got worse 
    melb_data2 = melb_data.dropna(axis=0)
    y2 = melb_data2.Price
    X2 = melb_data2.select_dtypes(exclude=["object"]).drop(["Price"],axis=1)
    X2_train, X2_valid, y2_train, y2_valid = train_test_split(X2, y2, test_size=0.2, train_size= 0.8, random_state=0)
    print(f"MAE = [{ score_prediction_accuracy(X2_train, X2_valid, y2_train, y2_valid)}]")

    # Approach 3 
    # drop columns with missing values 
    # MAE = 183,550 -- still worse than doing nothing. 
    melb_data3 = melb_data.dropna(axis=1)
    y3 = melb_data3.Price
    X3 = melb_data3.select_dtypes(exclude=["object"]).drop(["Price"],axis=1)
    X3_train, X3_valid, y3_train, y3_valid = train_test_split(X3, y3, test_size=0.2, train_size= 0.8, random_state=0)
    print(f"MAE = [{ score_prediction_accuracy(X3_train, X3_valid, y3_train, y3_valid)}]")

    # Approach 4 
    # use simple imputer 
    # MAE = 178,166 -- best till now 
    simpleImputer = SimpleImputer() 
    # Calculate the imputed values
    imputed_X_train = pd.DataFrame( simpleImputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame ( simpleImputer.transform(X_valid))
    # Fix the names of the columns 
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns
    print(f"Approach #4 : Imputed MAE = [{ score_prediction_accuracy(imputed_X_train, imputed_X_valid, y_train, y_valid)}]")

    



