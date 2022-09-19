''' Objective Function -> g(z1, z2) = (a - z1)^2 + b * (z2 - z1^2)^2

# Aim: To minimize the objective function

# Partial Derivatives:

    --> d/dz1(g) = -2*(a - z1) - 4 * b * z1(z2 - z1^2)
    --> d/dz2(g) = 2b (z2 - z1^2)
     
'''
# partial derivative w.r.t z1
def partial_z1(a,b, z1,z2):
    partial = (-2*(a-z1))-(4*b*z1*(z2-(z1**2)))
    return partial

# partial derivative w.r.t z2
def partial_z2(a,b, z1,z2):    
    partial = 2*b*(z2-z1**2)
    return partial

# Objective function value
def g_z1_z2(a,b, z1,z2):
    g=(a-z1)**2 + b*(z2-(z1**2))**2
    return g

# Update z1, z2 values
def update_z1_z2(z1,z2, lr, gz1, gz2):
    z1_upd=z1-(lr*gz1)
    z2_upd=z2-(lr*gz2)
    return z1_upd,z2_upd

# y = g(z1, z2)
y_values=[]

# initial a and b
a=2
b=10

# initial z1, z2

z1=5
z2=30
z1_vals=[z1]
z2_vals=[z2]

# learning rate
lr_vals=[0.0001]

# Calculate g(z1,z2) for initial values
y_values.append(g_z1_z2(a,b,z1,z2))

# Calculate Partial derivatives for initial values
partial_g_z1=partial_z1(a,b,z1,z2)
partial_g_z2=partial_z2(a,b,z1,z2)
partialz1=[partial_g_z1]
partialz2=[partial_g_z2]

for i in range(1,50):
    
    # Update z1 and z2 values for each iteration 
    z1,z2=update_z1_z2(z1,z2, lr_vals[0], partial_g_z1,partial_g_z2)
    
    # Storing those values for finding optimal values
    z1_vals.append(z1)
    z2_vals.append(z2)
    
    # Calculate g(z1,z2)
    y_values.append(g_z1_z2(a,b,z1,z2))
    
    #Calculate partial derivatives
    partial_g_z1=partial_z1(a,b,z1,z2)
    partial_g_z2=partial_z2(a,b,z1,z2)
    partialz1.append(partial_g_z1)
    partialz2.append(partial_g_z2)
    
# Finding minimum value of g(z1,z2)
min_y=min(y_values)
index_min_y=y_values.index(min_y)

# Finding optimal value of z1, z2
z1_opt=z1_vals[index_min_y]
z2_opt=z2_vals[index_min_y]

# Table for displaying all the values
from prettytable import PrettyTable
  
columns = ["Iter", "z1", "z2", "Partial z1", "Partial z2", "g(z1,z2)"]
myTable = PrettyTable()
# Add Columns
myTable.add_column(columns[0], range(50))
myTable.add_column(columns[1], z1_vals)
myTable.add_column(columns[2], z2_vals)
myTable.add_column(columns[3], partialz1)
myTable.add_column(columns[4], partialz2)
myTable.add_column(columns[5], y_values)
print(myTable)

# Plotting g(z1,z2) for each iteration
import matplotlib.pyplot as plt
import numpy as np
xpoints=range(50)
ypoints=y_values
plt.title(f"g(z1,z2) with learning rate={lr_vals[0]}\n (a,b)=({a,b}) \n initial (z1,z2)=({z1_vals[0],z2_vals[0]}\n Optimal (z1*,z2*)=({z1_opt,z2_opt})")
plt.plot(xpoints,ypoints)
plt.xlabel("Iterations")
plt.ylabel("g(z1,z2)")
plt.show()



