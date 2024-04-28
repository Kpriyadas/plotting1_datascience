#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
sales=[10,10,10,10,10]
year=[2020,2021,2022,2023,2024]
plt.plot(year,sales)


# In[2]:


x=[30,30,30,30,30]
y=[10,20,30,40,50]
plt.plot(x,y)


# In[3]:


x=[1,2,3,4,5]
y=[6,3,2,4,1]
plt.plot(x,y)


# In[4]:


plt.plot(year,gdp,color='green',marker='o',linestyle='solid')
plt.title("nomial gdp")
plt.ylabel("million of Rs.")
plt.legend("GDP")


# In[8]:


x = [10,20,30,40,50]
y = [30,30,30,30,30]
# plot lines
plt.plot(x, y, label = "line 1")
plt.plot(y, x, label = "line 2") 
plt.legend()


# In[10]:


str_input = input("Enter a string: ")
count = 0
for char in str_input:
    if char.isalpha():
        count += 1
print(count)


# In[11]:


import matplotlib.pyplot as plt
import numpy as np
#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])
plt.subplot(2, 2, 2)
plt.plot(x,y)
#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])
plt.subplot(2, 2, 2)
plt.plot(x,y)
plt.show()


# In[18]:


import matplotlib.pyplot as plt
import numpy as np
x=np.array([1,2,7,9])
y=np.array([2,4,1,9])
plt.subplot(2,1,2)
plt.plot(x,y)
plt.show()
#plot second
x=np.array([2,9,4,2])
y=np.array([3,8,9,8])
plt.subplot(2,1,2)
plt.plot(x,y)
plt.show()


# In[14]:


import matplotlib.pyplot as plt
import numpy as np
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])
plt.subplot(2, 3, 1)
plt.plot(x,y)
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])
plt.subplot(2, 3, 2)
plt.plot(x,y)
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])
plt.subplot(2, 3, 3)
plt.plot(x,y)
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])
plt.subplot(2, 3, 4)
plt.plot(x,y)
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])
plt.subplot(2, 3, 5)
plt.plot(x,y)
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])
plt.subplot(2, 3, 6)
plt.plot(x,y)
plt.show()


# In[9]:


import matplotlib.pyplot as plt
import numpy as np
x=np.array([1,4,2,5])
y=np.array([3,2,5,9])
plt.subplot(1,2,1)
plt.plot(x,y)
x=np.array([2,4,1,8])
y=np.array([2,3,1,8])
plt.subplot(1,2,2)
plt.plot(x,y)
x=np.array([2,5,8,5])
y=np.array([3,4,2,1])
plt.subplot(1,2,3)
plt.plot(x,y)
x=np.array([2,5,1,8])
y=np.array([3,2,1,5])
plt.subplot(1,2,4)
plt.plot(x,y)
plt.show()


# In[7]:


import matplotlib.pyplot as plt
import numpy as np

# First subplot
x1 = np.array([1, 4, 2, 5])
y1 = np.array([3, 2, 5, 9])
plt.subplot(1, 2, 1)
plt.plot(x1, y1, label='Subplot 1')
plt.title('Subplot 1')

# Second subplot
x2 = np.array([2, 4, 1, 8])
y2 = np.array([2, 3, 1, 8])
plt.subplot(1, 2, 2)
plt.plot(x2, y2, label='Subplot 2')
plt.title('Subplot 2')

plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-10,10,100)
y=np.sin(x)
plt.plot(x,y,marker="*")


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
plt.subplot(1,1,1)
x=np.linspace(-10,10,100)
y=np.cos(x)
plt.plot(x,y,marker="*",color="blue")
plt.subplot(1,1,1)
m=np.linspace(-20,20,100)
n=np.tan(m)
plt.plot(m,n,marker="*",color="orange")
plt.tight_layout()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-10,10,100)
y=np.tan(x)
plt.plot(x,y,marker="*")
m=np.linspace(-10,10,100)
n=np.cot(m)
plt.plot(m,n,marker="*")


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt
x=[1,2,3,4,5,6]
y=[1,4,9,16,25,36]
sns.scatterplot(x = x, y = y, color = 'red')
plt.xlabel('Numbers')
plt.ylabel('Square of Numbers')
plt.title('My First Graph')
plt.show()


# In[1]:


import matplotlib.pyplot as plt
# create data
sales = [10,10,10,10,10]
year = [2020,2021,2022,2023,2024]
# plot line
plt.plot(year, sales)


# In[ ]:




