import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'Python projects/Olympics_2024_predictor/Medals.xlsx'
df = pd.read_excel(file_path)

# Display the first few rows and the column names of the dataset
print(df.head())
print(df.columns)

# Rename columns for consistency
df.rename(columns={'Team/NOC': 'Country'}, inplace=True)

# Create 'Total_Medals' column
df['Total_Medals'] = df['Gold'] + df['Silver'] + df['Bronze']

# Create hypothetical GDP and Investment data
# In practice, you would use actual data
np.random.seed(42)
df['GDP'] = np.random.uniform(1000, 50000, len(df))
df['Investment'] = np.random.uniform(100, 5000, len(df))

# Sum GDP and Investment to create a new feature
df['GDP_Investment'] = df['GDP'] + df['Investment']

# Display the preprocessed dataframe
print(df.head())

# Group by country to get total medals and other features
country_medals = df.groupby('Country')[['Total_Medals', 'GDP_Investment']].sum().reset_index()

# Visualize the total medals by country
plt.figure(figsize=(15, 10))
sns.barplot(x='Total_Medals', y='Country', data=country_medals.sort_values(by='Total_Medals', ascending=False).head(20))
plt.title('Top 20 Countries by Total Medals in 2021 Tokyo Olympics')
plt.xlabel('Total Medals')
plt.ylabel('Country')
plt.show()

# Prepare data for training
X = country_medals[['GDP_Investment']]
y = country_medals['Total_Medals']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Generate hypothetical GDP and Investment data for 2024
np.random.seed(2024)
future_gdp_investment = np.random.uniform(1000, 50000, len(country_medals))

# Create a DataFrame for future predictions
future_data = pd.DataFrame({
    'Country': country_medals['Country'],
    'GDP_Investment': future_gdp_investment
})

# Predict the total medals for 2024
future_data['Predicted_Total_Medals'] = model.predict(future_data[['GDP_Investment']])

# Display the predictions
print(future_data.sort_values(by='Predicted_Total_Medals', ascending=False).head(20))

# Visualize the predictions
plt.figure(figsize=(15, 10))
sns.barplot(x='Predicted_Total_Medals', y='Country', data=future_data.sort_values(by='Predicted_Total_Medals', ascending=False).head(20))
plt.title('Predicted Top 20 Countries by Total Medals in 2024 Olympics')
plt.xlabel('Predicted Total Medals')
plt.ylabel('Country')
plt.show()
