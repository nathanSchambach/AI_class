import math

#dataset
x1 = [0,0,1,1]
x2 = [0,1,0,1]
y = [0,1,1,0]

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

#set intitial weights
w0_3 = -0.5
w1_3 = -0.4
w2_3 = -0.3
w0_4 = -0.2
w1_4 = 0
w2_4 = 0.2
w0_5 = 0.3
w3_5 = 0.4
w4_5 = 0.5

print('Intiial weights: \n', w0_3, w1_3, w2_3, "\n",
w0_4, w1_4, w2_4, "\n",
w0_5, w3_5, w4_5)

iteration = 0
max_iterations = 10000000
alpha = 0.05
tolerance = 1e-10

#ANN
while iteration < max_iterations:
    iteration += 1
    
    prev_weights = [w0_3, w1_3, w2_3, w0_4, w1_4, w2_4, w0_5, w3_5, w4_5]

    index = iteration % 4

    #forward pass
    in3 = w0_3 + w1_3 * x1[index] + w2_3 * x2[index]
    a3 = sigmoid(in3)

    in4 = w0_4 + w1_4 * x1[index] + w2_4 * x2[index]
    a4 = sigmoid(in4)

    in5 = w0_5 + w3_5 * a3 + w4_5 * a4
    a5 = sigmoid(in5)

    #backpropagation
    delta5 = (y[index] - a5) * sigmoid_derivative(a5)
    delta3 = sigmoid_derivative(a3) * w3_5 * delta5
    delta4 = sigmoid_derivative(a4) * w4_5 * delta5

    #weight update
    w0_3 += alpha * 1 * delta3
    w1_3 += alpha * x1[index] * delta3
    w2_3 += alpha * x2[index] * delta3

    w0_4 += alpha * 1 * delta4
    w1_4 += alpha * x1[index] * delta4
    w2_4 += alpha * x2[index] * delta4

    w0_5 += alpha * 1 * delta5
    w3_5 += alpha * a3 * delta5
    w4_5 += alpha * a4 * delta5

    #check for convergence
    converged = True
    for i in range(len(prev_weights)):
        if abs(prev_weights[i] - [w0_3, w1_3, w2_3, w0_4, w1_4, w2_4, w0_5, w3_5, w4_5][i]) > tolerance:
            converged = False
            break

    if converged:
        break
print(iteration)
print('final weights: \n',
      w0_3, w1_3, w2_3, "\n",
        w0_4, w1_4, w2_4, "\n",
            w0_5, w3_5, w4_5)

#create predict
def predict(input_x1, input_x2):
    in3 = w0_3 + w1_3 * input_x1 + w2_3 * input_x2
    a3 = sigmoid(in3)

    in4 = w0_4 + w1_4 * input_x1 + w2_4 * input_x2
    a4 = sigmoid(in4)

    in5 = w0_5 + w3_5 * a3 + w4_5 * a4
    a5 = sigmoid(in5)
    return a5

#test the model
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
predictions = []

for input_x1, input_x2 in test_inputs:
    prediction = predict(input_x1, input_x2)
    predictions.append(round(prediction))

print("Predictions:", predictions)

#compare predictions
correct_predictions = 0
for i in range(len(y)):
    if predictions[i] == y[i]:
        correct_predictions += 1

#calculate accuracy 
accuracy = correct_predictions / len(y)
print("Accuracy:", accuracy)
