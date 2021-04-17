# ### Predict the percentage of an student based on the no. of study hours.

#importing libraries
import pandas as pd  
import numpy as np    
import matplotlib.pyplot as plt 
import seaborn as sns

# Reading data
url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("Data imported successfully")

df.head(5)


# ## Describing Data

df.info()

df.describe()

#checking for null values
print(df.isnull().sum())


# ## Plotting Data

#Plotting the distribution of scores
df.plot(x='Hours', y='Scores', style='o',color='black')    
plt.title('Hours vs Percentage')    
plt.xlabel('Hours Studied')    
plt.ylabel('Percentage Score')    
plt.show()  

#countplot for hours
sns.countplot(x='Hours',data = df)

#distribution plot for Hours column
sns.set_style('whitegrid')
sns.distplot(df['Hours'], kde = True, color ='red', bins = 20)


# ## Preparing Data

#separating dataset into inputs and target variable
inputs = df.iloc[:, :-1].values    
target = df.iloc[:, 1].values  


# ## Splitting Dataset into Train Test

#train-test split
from sklearn.model_selection import train_test_split    
X_train, X_test, y_train, y_test = train_test_split(inputs, target,test_size=0.2, random_state=0)   


# ## Training Algorithm

# ### Using Linear Regression 

from sklearn.linear_model import LinearRegression    
regressor = LinearRegression()    
regressor.fit(X_train, y_train)   
  
print("Training ... Completed !.")  

line = regressor.coef_*inputs+regressor.intercept_  
plt.scatter(inputs, target)  
plt.plot(inputs, line,color='black');  
plt.show() 


# ## Predictions

y_pred = regressor.predict(X_test)  

#plotting y_test and y_pred
plt.scatter(y_test,y_pred)


# ## Comparing Actual Values and Predicted Values 

pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})    
pred 


# ## Predicting percentage of students for 9.25 studying hours 

#predicting percentage for 9.25 hours of studying
hours = [[9.25]]  
own_pred = regressor.predict(hours)  
print("Number of hours = {}".format(hours))  
print("Prediction Score = {}".format(own_pred[0]))  


# # Evaluating performance of algorithm

from sklearn import metrics  

print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))

print('Root Mean Squared Error:', 
      np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

