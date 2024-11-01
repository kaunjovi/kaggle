import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def score_prediction_accuracy(X_train, X_valid, y_train, y_valid, message = "MAE =") : 
    rt_model = RandomForestRegressor( n_estimators= 10, random_state=0)
    rt_model.fit( X_train, y_train)
    predictions = rt_model.predict(X_valid)
    mae = mean_absolute_error( predictions, y_valid)
    print(f"{message} {mae}")
    return mae

def score_prediction_accuracy_use_data_with_missing_values(melb_data) : 
    y = melb_data.Price

    X = melb_data.select_dtypes(exclude=["object"]).drop(["Price"],axis=1)

    # print(f"X = {X.head()}")
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, train_size= 0.8, random_state=0)

    score_prediction_accuracy(X_train, X_valid, y_train, y_valid, "Approach # 1 : With missing data. MAE = ")


def score_prediction_accuracy_drop_missing_rows ( melb_data) : 
    melb_data2 = melb_data.dropna(axis=0)
    y2 = melb_data2.Price
    X2 = melb_data2.select_dtypes(exclude=["object"]).drop(["Price"],axis=1)
    X2_train, X2_valid, y2_train, y2_valid = train_test_split(X2, y2, test_size=0.2, train_size= 0.8, random_state=0)
    # print(f"Approach # 2 : Dropped rows MAE = [{ score_prediction_accuracy(X2_train, X2_valid, y2_train, y2_valid)}]")
    # print(f"Approach # 2 : Dropped rows MAE = [{ score_prediction_accuracy(X2_train, X2_valid, y2_train, y2_valid)}]")
    score_prediction_accuracy(X2_train, X2_valid, y2_train, y2_valid, "Approach # 2 : Dropped rows MAE = " ) 

def score_prediction_accuracy_drop_missing_columns( melb_data ) : 
    melb_data3 = melb_data.dropna(axis=1)
    y3 = melb_data3.Price
    X3 = melb_data3.select_dtypes(exclude=["object"]).drop(["Price"],axis=1)
    X3_train, X3_valid, y3_train, y3_valid = train_test_split(X3, y3, test_size=0.2, train_size= 0.8, random_state=0)
    # print(f"Approach # 3 : Dropped columns MAE = [{ score_prediction_accuracy(X3_train, X3_valid, y3_train, y3_valid)}]")
    score_prediction_accuracy(X3_train, X3_valid, y3_train, y3_valid, "Approach # 3 : Dropped columns MAE = ")

def score_prediction_accuracy_imputation( melb_data ) : 
    y = melb_data.Price
    X = melb_data.select_dtypes(exclude=["object"]).drop(["Price"],axis=1)

    # print(f"X = {X.head()}")
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, train_size= 0.8, random_state=0)

    simpleImputer = SimpleImputer() 
    # Calculate the imputed values
    imputed_X_train = pd.DataFrame( simpleImputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame ( simpleImputer.transform(X_valid))
    # Fix the names of the columns 
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns
    # print(f"Approach # 4 : Imputed MAE = [{ score_prediction_accuracy(imputed_X_train, imputed_X_valid, y_train, y_valid)}]")
    score_prediction_accuracy(imputed_X_train, imputed_X_valid, y_train, y_valid, "Approach # 4 : Imputed MAE = ")

def exploratory_data_analysis (melb_data) : 
    
    # Shape = [(13580, 21)]
    print(f"Shape = [{ melb_data.shape}]")

    # missing_val_count_by_column = melb_data.isnull().sum() 
    missing_val_count_by_column = melb_data.isnull().sum()
    # print(f"Missing value count by column \n{missing_val_count_by_column}")
    print(f"Name of columns with missing values \n{ missing_val_count_by_column[missing_val_count_by_column > 0]}")
    # Name of columns with missing values 
    # Car               62
    # BuildingArea    6450
    # YearBuilt       5375
    # CouncilArea     1369


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

def score_prediction_accuracy_imputation_plus (melb_data) : 

    y = melb_data.Price
    X = melb_data.select_dtypes(exclude=["object"]).drop(["Price"],axis=1)
    # print(f"X = {X.head()}")
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, train_size= 0.8, random_state=0)

    X4_train = X_train.copy()
    X4_valid = X_valid.copy() 
    # print(f"{ X4_train.head() }")

    # Get names of columns with missing values
    cols_with_missing = [ col for col in X_train.columns if X_train[col].isnull().any()]
    # print(f"Columns with missing values {cols_with_missing}")

    # Add another column mentioning the rows that will be imputed. 
    for col in cols_with_missing : 
        X4_train[col + "_isnull"] = X4_train[col].isnull() 
        X4_valid[col + "_isnull"] = X4_valid[col].isnull() 

    # print(f"{ X4_train.head() }")

    simpleImputer = SimpleImputer() 

    # Calculate the imputed values
    imputed_X4_train = pd.DataFrame( simpleImputer.fit_transform(X4_train))
    imputed_X4_valid = pd.DataFrame ( simpleImputer.transform(X4_valid))

    imputed_X4_train.columns = X4_train.columns
    imputed_X4_valid.columns = X4_valid.columns 

    # print(f"{ imputed_X4_train.head() }")
    # print(f"{ imputed_X4_valid.head() }")
    
    score_prediction_accuracy( imputed_X4_train, imputed_X4_valid, y_train, y_valid, "Approach # 5 : Imputer plus. MAE =")

if __name__ == "__main__" : 
    print(f"Panda version [{pd.__version__}]")
    pd.set_option('display.max_rows', None, 'display.max_columns', None)

    # Load data. There is no index coln. 
    melb_data = pd.read_csv("data/intermediate-ml/melb_data.csv" )
    # exploratory_data_analysis(melb_data)

    score_prediction_accuracy_use_data_with_missing_values(melb_data)
    score_prediction_accuracy_drop_missing_rows( melb_data )
    score_prediction_accuracy_drop_missing_columns( melb_data )
    score_prediction_accuracy_imputation( melb_data )
    score_prediction_accuracy_imputation_plus(melb_data) 

