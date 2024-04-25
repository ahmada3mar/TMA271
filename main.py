
from ucimlrepo import fetch_ucirepo 
from table import Table

# Please install the requirements by run : pip install -r .\requirements.txt
# To run the code run: python .\main.py


# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 

# prepare the table 
table = Table(heart_disease.data)

# Question 1:
# To display the first and last 10 rows
print("\n\n\n=========== Start Question 1 ===============")
table.display_rows(10) # first 10
table.display_rows(10,reverse=True) # last 10
print("=========== End Question   1 ===============")

#================================


# Question 2:

# Display basic statistics of the dataset
print("\n\n\n=========== Start Question 2 ===============")
table.statistics()
#Find any missing values
table.display_missing_values()
print("=========== End Question   2 ===============")


# Question 3
print("\n\n\n=========== Start Question 3 ===============")
# fill missing data
table.display_missing_values()
# data preprocessing
table.handle_missing_values()
print("missing data fixed and the result is")

# Display data after fixing
table.display_missing_values()
print("=========== End Question   3 ===============")

#================================


# Question 4
# split data into 80% training and 20% testing
print("\n\n\n=========== Start Question 4 ===============")
table.splitData()
print("=========== End Question   4 ===============")

#=====================



# Question 5
#  Choose and implement a regression
print("\n\n\n=========== Start Question 5 ===============")
table.build_model()
print("=========== End Question   5 ===============")

#=====================



# Question 6
#  Evaluate the model's performance
print("\n\n\n=========== Start Question 6 ===============")
table.evaluate()
print("=========== End Question   6 ===============")

#=====================


# Question 7
#  Evaluate the model's performance
print("\n\n\n=========== Start Question 7 ===============")
table.fine_tune()
print("=========== End Question   7 ===============")

#=====================

print("\n\nBy Ahmad E'mar", "Number: 2220801" , sep="\n")





