# https://www.kaggle.com/code/kaunjovi/exercise-underfitting-and-overfitting/edit 

# there is a sweet spot between underfitting and overfitting where MAE is the least. 
# we want to train our model with the whole data in that sweet spot. 

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


def get_mae_for_max_leaf_nodes(max_leaf_nodes, train_X, val_X, train_y, val_y) : 

    iowa_model = DecisionTreeRegressor(max_leaf_nodes= max_leaf_nodes, random_state=1)
    # Fit Model
    iowa_model.fit(train_X, train_y)

    # Make validation predictions and calculate mean absolute error
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print(f"max leaf nodes [{max_leaf_nodes}] # mae [{ val_mae}]")


if __name__ == "__main__" : 
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

    for max_leaf_nodes in [5, 25, 50, 100, 250, 500, 5000] : 
        get_mae_for_max_leaf_nodes(max_leaf_nodes, train_X, val_X, train_y, val_y)


# max leaf nodes [5] # mae [35044.51299744237]
# max leaf nodes [25] # mae [29016.41319191076]
# max leaf nodes [50] # mae [27405.930473214907]
# max leaf nodes [100] # mae [27282.50803885739] -- sweety 
# max leaf nodes [250] # mae [27430.850744944964]
# max leaf nodes [500] # mae [28380.917944156296]
# max leaf nodes [5000] # mae [29001.372602739724]
