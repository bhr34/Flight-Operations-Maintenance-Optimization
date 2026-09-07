import pandas as pd
import zipfile

zip_path = r"C:\Users\Asus\Downloads\flights.csv.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # List the CSV file in the ZIP file
    file_names = zip_ref.namelist()
    print("Files in ZIP:", file_names)
    
    
    with zip_ref.open(file_names[0]) as file:
        df = pd.read_csv(file)
        print("File loaded successfully! First 5 rows:")
        print(df.head())
print("All column names:")
print(df.columns)
print("\n Missing value check:")
print(df.isnull().sum())

df_clean = df.dropna(subset=['dep_delay', 'arr_delay'])

print(f"Cleaned dataset size: {df_clean.shape}")
import matplotlib.pyplot as plt

hourly_delay = df_clean.groupby('hour')['dep_delay'].mean()


plt.figure(figsize=(10,6))
hourly_delay.plot(kind='bar', color='salmon')
plt.title("Average Departure Delay by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Delay (min)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

monthly_delay = df.groupby("month")["dep_delay"].mean()
plt.figure(figsize=(10, 5))
monthly_delay.plot(kind="bar", color="steelblue")
plt.title("Average Departure Delay by Month")
plt.xlabel("Month")
plt.ylabel("Average Delay (minutes)")
plt.grid(True)
plt.tight_layout()
plt.show()

airline_delay = df.groupby("name")["dep_delay"].mean().sort_values()
plt.figure(figsize=(10, 7))
airline_delay.plot(kind="barh", color="salmon")
plt.title("Average Departure Delay by Airline")
plt.xlabel("Average Delay (minutes)")
plt.ylabel("Airline")
plt.grid(True)
plt.tight_layout()
plt.show()
import pandas as pd
import zipfile

zip_path = r"C:\Users\Asus\Downloads\flights.csv.zip"

with zipfile.ZipFile(zip_path, 'r') as z:
    
    print("ZIP içindeki dosyalar:", z.namelist())
    
   
    with z.open("flights.csv") as f:
        df = pd.read_csv(f)


print("CSV uploaded successfully . First 5 rows:")
print(df.head())
import matplotlib.pyplot as plt


df_clean = df.dropna(subset=['arr_delay'])


avg_delays = df_clean.groupby("name")["arr_delay"].mean().sort_values()


plt.figure(figsize=(10,6))
avg_delays.plot(kind="barh", color="skyblue")
plt.xlabel("Average Arrival Delay (minutes)")
plt.title("Average Arrival Delay by Airline")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 

df_delay = df[['dep_delay', 'arr_delay']].dropna()

correlation = df_delay['dep_delay'].corr(df_delay['arr_delay'])
print(f"📈 Correlation between Departure and Arrival Delay: {correlation:.2f}")


plt.figure(figsize=(8,6))
sns.scatterplot(data=df_delay.sample(1000), x='dep_delay', y='arr_delay', alpha=0.3, color='purple')
plt.title("Departure Delay vs. Arrival Delay")
plt.xlabel("Departure Delay (min)")
plt.ylabel("Arrival Delay (min)")
plt.grid(True)
plt.tight_layout()
plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


df_model = df[['dep_delay', 'arr_delay']].dropna()


X = df_model[['dep_delay']]
y = df_model['arr_delay']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
print(f"R² Score: {r2:.2f}")


plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='gray', alpha=0.4, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted Line')
plt.title("Arrival Delay Prediction")
plt.xlabel("Departure Delay (minutes)")
plt.ylabel("Arrival Delay (minutes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()








