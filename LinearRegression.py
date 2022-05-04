import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Dataframe display setup
desired_width = 410
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)

customers = pd.read_csv('Ecommerce Customers')
print(customers.head())

# Data Visualization
# sns.jointplot(x=customers['Time on Website'], y=customers['Yearly Amount Spent'])
# sns.jointplot(x=customers['Time on App'], y=customers['Yearly Amount Spent'])
# sns.jointplot(x=customers['Time on App'], y=customers['Length of Membership'], kind='hex')
# sns.pairplot(customers)
# sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)

# Model Training & Testing
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

# Creating model instance
lm = LinearRegression()
# Training/Fitting on Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm.fit(X_train, y_train)
print(lm.coef_)

predictions = lm.predict(X_test)
# sns.scatterplot(x=y_test, y=predictions)
# plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

sns.displot((y_test-predictions), bins=50, kde=True);
# plt.show()

# Analyzing coefficients to determine if we should focus our efforts on the website or app
coeffecients = pd.DataFrame(lm.coef_, X.columns)
coeffecients.columns = ['Coeffecient']
print(coeffecients)
