# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# import joblib

# data = pd.read_csv('https://raw.githubusercontent.com/VarunPalrecha/DataSets/main/output1.csv')

# X = data[['category']]
# y = data['price']

# column_transformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# X_encoded = column_transformer.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# model = LinearRegression()
# model.fit(X_train, y_train)

# # Save the trained model and column transformer for later use
# joblib.dump(column_transformer, 'column_transformer.pkl')
# joblib.dump(model, 'linear_regression_model.pkl')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

def perform_function(input_data):
    data = pd.read_csv('https://raw.githubusercontent.com/VarunPalrecha/DataSets/main/output1.csv')
    X = data[['category']]
    y = data['price']
    column_transformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    X_encoded = column_transformer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    new_category = [[input_data]]  # Example new category value
    new_category_encoded = column_transformer.transform(new_category)
    predicted_price = model.predict(new_category_encoded)
    return f'Processed: {predicted_price[0]}'
