import pandas as pd

# Define inputs, weights and labels
weights = [1, 2]
inputs = [(0,0), (0,1), (1,0), (1,1)]
labels = [0, 0, 0, 1]
bias = -3
outputs = []

# Define Net Sum
for input, label in zip(inputs, labels):
	linear_combination = weights[0] * input[0] + weights[1] * input[1] + bias
	out = int(linear_combination >=0)   # Activation output
	outputs.append([input[0], input[1], linear_combination, out])

out_frame = pd.DataFrame(outputs , columns=['Input 1', 'Input 2', 'Linear Combination', 'Output'])
print(out_frame.to_string(index=False))





