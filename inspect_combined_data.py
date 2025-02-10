import pandas as pd

# Load combined data files
user_data = pd.read_csv("Combined_data\combined_user_data.csv")
user_activities = pd.read_csv("Combined_data\combined_user_activities.csv")
user_reviews = pd.read_csv("Combined_data\combined_user_reviews.csv")

# Display basic information
print("User Data Info:")
print(user_data.info())
print("\nUser Activities Info:")
print(user_activities.info())
print("\nUser Reviews Info:")
print(user_reviews.info())

# Show sample rows
print("\nSample User Data:")
print(user_data.head())
print("\nSample User Activities:")
print(user_activities.head())
print("\nSample User Reviews:")
print(user_reviews.head())
