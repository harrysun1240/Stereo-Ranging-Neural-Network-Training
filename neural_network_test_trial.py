import random
import csv
import pandas as pd
import statistics
from sklearn.neural_network import MLPRegressor

x_range = (0, 11000)
y_range = (5000, 15000)
filename = 'coordinates.csv'
output_filename = 'dataset.csv'
target_column = ['L']
predictors = ['R1', 'R2']
max_distance = 15000

def generate_coordinates(num_coordinates, x_range, y_range):
    coordinates = []
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    for _ in range(num_coordinates):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        coordinates.append((x, y))
    
    return coordinates

def export_coordinates_to_csv(coordinates, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])
        writer.writerows(coordinates)


def calculate_R1(x, y):
    focal_length = 50
    return focal_length * (x - 5000) / y

def calculate_R2(x, y):
    focal_length = 50
    return focal_length * (6000 - x) / y

def process_coordinates(input_filename, output_filename):
    with open(input_filename, 'r') as input_file, open(output_filename, 'w', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)
        
        header = next(reader)
        
        writer.writerow(['R1', 'R2', 'L'])
        
        for row in reader:
            x = float(row[0])
            y = float(row[1])
            
            R1 = calculate_R1(x, y)
            R2 = calculate_R2(x, y)
            
            writer.writerow([R1, R2, y])

coordinates = generate_coordinates(100000, x_range, y_range)
export_coordinates_to_csv(coordinates, filename)
print(f"Coordinates exported to '{filename}' successfully.")
process_coordinates(filename, output_filename)
print(f"Data exported to '{output_filename}' successfully.")

df = pd.read_csv('dataset.csv')

df[predictors] += 50

df[predictors] = df[predictors] / df[predictors].max()
df[target_column] = df[target_column] / max_distance

test_data = df.sample(frac=0.3)
train_data = df.drop(test_data.index)

x_train = train_data[predictors].values
y_train = train_data[target_column].values.ravel()
x_test = test_data[predictors].values
y_test = test_data[target_column].values.ravel()

mlp = MLPRegressor(hidden_layer_sizes=(6,), activation='relu', solver='adam', max_iter=10000)
mlp.fit(x_train, y_train)

predict_test = mlp.predict(x_test)
difference = predict_test * max_distance - y_test * max_distance

result_df = pd.DataFrame({
    'Original Index': test_data.index,
    'Actual Value': y_test * max_distance,
    'Predicted Value': predict_test * max_distance,
    'Difference': difference
})

mean = abs(statistics.mean(difference))

print("Mean: ", mean)

result_df = result_df.sort_values(by='Original Index')

result_df.to_csv('predicted_values.csv', index=False)

final_filename = 'predicted_values.csv'
print(f"Data exported to '{final_filename}' successfully.")