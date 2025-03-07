#simple data for excel data
import pandas as pd
import os
import sys

# Step 1: Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

df = pd.DataFrame(data)

# Step 2: Save DataFrame to Excel
file_path = 'output.xlsx'
df.to_excel(file_path, index=False)

# Step 3: Open the Excel file automatically
if sys.platform == "win32":  # For Windows
    os.startfile(file_path)
elif sys.platform == "darwin":  # For macOS
    os.system(f"open {file_path}")
else:  # For Linux
    os.system(f"xdg-open {file_path}")




#example data for plotting
import matplotlib.pyplot as plt

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a line plot
plt.bar(x, y)

# Add title and labels
plt.title("Line Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Show the plot
plt.show()



#example code for bar chart with label
import matplotlib.pyplot as plt

# Example data
categories = ['A', 'B', 'C', 'D']
values = [10, 20, 15, 30]

# Create a bar plot
plt.bar(categories, values)

# Add title and labels
plt.title("Bar Plot Example")
plt.xlabel("Categories")
plt.ylabel("Values")

# Show the plot
plt.show()



#example code for histogram
import matplotlib.pyplot as plt

# Example data
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

# Create a histogram
plt.hist(data, bins=5, color='skyblue', edgecolor='black')

# Add title and labels
plt.title("Histogram Example")
plt.xlabel("Value")
plt.ylabel("Frequency")

# Show the plot
plt.show()


#example code for scatter plot
import matplotlib.pyplot as plt

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a scatter plot
plt.scatter(x, y)

# Add title and labels
plt.title("Scatter Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Show the plot
plt.show()




#example code for count plot
import seaborn as sns
import matplotlib.pyplot as plt

# Example data: Titanic dataset
data = sns.load_dataset("titanic")

# Create a count plot
sns.countplot(x="class", data=data)

# Add title
plt.title("Count Plot of Titanic Classes")

# Show the plot
plt.show()



#example code for customization in the plot
import matplotlib.pyplot as plt

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a line plot with customization
plt.plot(x, y, color='green', linestyle='--', marker='o')

# Customize the plot
plt.title("Customized Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)

# Show the plot
plt.show()



#this is the code for the physics problem 
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dt=np.arange(0,17.65,0.01)
g=9.8 
Vo=100  
xo=0
theta=(60*3.14)/180  
Vox=Vo*np.cos(theta)
x=xo+(Vox*dt)
trajectory=(x*np.tan(theta))-((g*x*2)/(2*Vox*2))
print(trajectory)
X=[]
Y=[]
for i in np.arange(len(x)):
    E=x[i]
    S=trajectory[i]
    X.append(E)
    Y.append(S)
    
p=pd.DataFrame(list(zip(X,Y)),columns=["Position","trajectory"])  
p.to_excel("projetile motion.xlsx",index=False)

if sys.platform == "win32":   
   os.startfile("projetile motion.xlsx") 


plt.plot(x,trajectory)
plt.xlabel("position",fontsize=20) 
plt.ylabel("trajectory",fontsize=20) 
plt.grid()
plt.show()




#try code
import pandas as pd
import matplotlib.pyplot as plt
x=[2,3,4,5,6,7,8,9]
y=[9,8,7,6,5,4,3,2]
plt.bar(x,y)
plt.title("barchart")
plt.grid(True)
plt.show()