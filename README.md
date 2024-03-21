import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from CSV
df = pd.read_csv('Housing.csv')
print(df.head())

print(df.isnull().sum())

X = df[['area','bedrooms','bathrooms','stories','parking']]
y= df['price']

sns.pairplot(df)

sns.heatmap(df.corr(), annot=True)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
mean_s_predict= mean_squared_error(y_test,y_pred)
r2_scored=r2_score(y_test,y_pred)
print("Mean Squared Eroor:",mean_s_predict)
print("r2_score:",r2_scored)

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()
