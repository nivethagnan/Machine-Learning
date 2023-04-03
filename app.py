from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# Load the data into a pandas dataframe
df = pd.read_csv('household_power_consumption.txt', delimiter=';',
                 parse_dates={'dt' :[0,1]}, infer_datetime_format=True,
                 na_values=['nan','?'], index_col='dt')

# Resample the data to hourly intervals
df = df.resample('1H').mean()

# Remove any missing values
df.dropna(inplace=True)

# Create a dataframe for the target variable (global_active_power)
y = df['Global_active_power'].values.reshape(-1,1)

# Create a dataframe for the predictor variables (voltage, global_reactive_power, etc.)
X = df.drop('Global_active_power', axis=1).values

# Split the data into training and testing sets
train_size = int(0.8 * len(df))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train a KNN model to predict the global active power consumption
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

# Use the trained KNN model to predict the global active power consumption
y_pred = knn.predict(X_test)

# Use the trained KNN model to disaggregate the global active power consumption into individual appliance consumption
individual_power = knn.predict(X)

# Add the disaggregated individual appliance consumption to the dataframe
df['Individual_appliance_power'] = individual_power

# Identify the predicted appliances that use the most energy
mean_energy = np.mean(y_pred, axis=1)

# Create dictionary to store mean energy consumption for each appliance
appliance_energy = {
    'sub_metering_1': mean_energy[0],
    'sub_metering_2': mean_energy[1],
    'sub_metering_3': mean_energy[2]
}

# Sort appliances by mean energy consumption in descending order
sorted_appliances = sorted(appliance_energy.items(), key=lambda x: x[1], reverse=True)

@app.route('/data/<appliance>')
def data(appliance):
    global sorted_appliances
    mean_energy_dict = dict(sorted_appliances)
    if appliance in mean_energy_dict:
        return jsonify(mean_energy_dict[appliance])
    else:
        return "Appliance not found"

@app.route('/', methods=['GET', 'POST'])
def home():
    global sorted_appliances
    cost_per_kwh = 0.0
    usage_hours = 0.0
    if request.method == 'POST':
        sub_metering_1 = requests.get('http://localhost:5000/data/sub_metering_1').text
        sub_metering_2 = requests.get('http://localhost:5000/data/sub_metering_2').text
        sub_metering_3 = requests.get('http://localhost:5000/data/sub_metering_3').text
        cost_per_kwh = float(request.form['cost_per_kwh'])
        usage_hours = float(request.form['usage_hours'])
        sub_metering_1_cost = round((float(sub_metering_1) * cost_per_kwh * usage_hours) / 100, 2)
        sub_metering_2_cost = round((float(sub_metering_2) * cost_per_kwh * usage_hours) / 100, 2)
        sub_metering_3_cost = round(float(sub_metering_3) * cost_per_kwh * usage_hours / 1009, 2)
        sub_metering_1_cost = requests.get('http://localhost:5000/data/sub_metering_1_cost').text
        sub_metering_2_cost = requests.get('http://localhost:5000/data/sub_metering_2_cost').text
        sub_metering_3_cost = requests.get('http://localhost:5000/data/sub_metering_3_cost').text
        return render_template('home.html', sub_metering_1=sub_metering_1, sub_metering_2=sub_metering_2, sub_metering_3=sub_metering_3, sub_metering_1_cost=sub_metering_1_cost, sub_metering_2_cost=sub_metering_2_cost, sub_metering_3_cost=sub_metering_3_cost)
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)