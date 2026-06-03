import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv('data/house_data.csv')

x = data[['area']]
y = data['price']

model = LinearRegression()
model.fit(x, y)

joblib.dump(model, 'model.pkl')
print('Model trained and saved as model.pkl')