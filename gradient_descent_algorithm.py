#HW2 - CSC481 - Nathan Schambach

#dataset
x = [2,4,6,7,8,10,9]
y = [5,7,14,14,17,19,20]

#previous w0 & w1 values
w0Prev = 0 
w1Prev = 0

#hold hw(x) values
h = [0,0,0,0,0,0,0]

#learning rate
alpha = 0.00001

#initial weights
w0 = 0.25
w1 = 0.25
iterations = 0

#Batch Gradient Descent Algorithm
for i in range(10000000):
    iterations += 1
    index = 0
    calc0 = 0
    calc1 = 0
    w0Prev = w0
    w1Prev = w1

    #hw(x) = w0 +w1*x
    for i in range(8):
        index = i%7
        h[index] = w0 + w1*x[index]

    for i in range(8):
        index = i%7
        calc0 = calc0 + (y[index]-h[index])
        calc1 = calc1 + (y[index]-h[index])*x[index]

    #update weights
    w0 = w0 + alpha*calc0
    w1 = w1 + alpha*calc1
    
    #Check for convergence
    if (abs(w0 - w0Prev)< (10**-10)) and (abs(w1 - w1Prev)< (10**-10)):
        break

print('Batch Gradient Descent')
print('# of Iterations: ' + str(iterations))
print('w0 = ' + str(w0))
print('w1 = ' + str(w1))

print('-------------------------------------')

#set to initial weights
w0 = 0.25
w1 = 0.25
iterations = 0

#Stochastic Gradient Descent Algorithm
for i in range(10000000):
    iterations += 1
    index = 0
    w0Prev = w0
    w1Prev = w1

    for i in range(8):
        index = i%7

        #hw(x) = w0 +w1*x
        h[index] = w0 + w1*x[index]
        
        calc0 = (y[index]-h[index])
        calc1 = (y[index]-h[index])*x[index]

        #update weights
        w0 = w0 + alpha*calc0
        w1 = w1 + alpha*calc1
    
    #Check for convergence
    if (abs(w0 - w0Prev)< (10**-10)) and (abs(w1 - w1Prev)< (10**-10)):
        break

print('Stochastic Gradient Descent')
print('# of Iterations: ' + str(iterations))
print('w0 = ' + str(w0))
print('w1 = ' + str(w1))


