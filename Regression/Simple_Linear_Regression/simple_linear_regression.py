#Simple Linear Regression

import numpy as np          
import matplotlib.pyplot as plt 
import pandas as pd

# Import dataset

dataset = pd.read_csv('Salary_data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#Split the data Set into Training Set and Test Set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)


#feature Scaling
##SLR takes care of feature Scaling

"""from sklearn.preprocessing import StandardScaler
sc_x  = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

# fit Simple Linear Regression to training Set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predict the test set results

y_pred = regressor.predict(x_test)

# Visualize the Training set results

plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience (Training set)')

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualize the Test set results

plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience (Test set)')

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



