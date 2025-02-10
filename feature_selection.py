import pandas as pd
import os


def feature_selection(
    user_data_path="Combined_data/combined_user_data.csv",
    user_activities_path="Combined_data/combined_user_activities.csv",
    user_reviews_path="Combined_data/combined_user_reviews.csv",
    output_folder="experiment_folder/selected_features",
):
    """
    Loads the three consolidated Meituan data files, selects relevant features for HPFL,
    and saves them to new CSV files.

    Args:
        user_data_path (str): Path to the consolidated user_data CSV file.
        user_activities_path (str): Path to the consolidated user_activities CSV file.
        user_reviews_path (str): Path to the consolidated user_reviews CSV file.
        output_folder (str): Directory where the selected-features CSV files will be saved.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Loading data...")
    user_data = pd.read_csv(user_data_path)
    user_activities = pd.read_csv(user_activities_path)
    user_reviews = pd.read_csv(user_reviews_path)
    print("Data loaded successfully!\n")

    print("Selecting relevant features...")

    # User Data: user_id, gender, city, age, contribution, tag
    user_data_features = ["user_id", "gender", "city", "age", "contribution", "tag"]
    user_data_selected = user_data[user_data_features]

    # User Activities: user_id, event_id, business_id, response, is_chosen, timestamp, participants_gender
    user_activities_features = [
        "user_id",
        "event_id",
        "business_id",
        "response",
        "is_chosen",
        "timestamp",
    ]
    user_activities_selected = user_activities[user_activities_features]

    # User Reviews: user_id, business_id, timestamp, text, rate, star
    user_reviews_features = [
        "user_id",
        "business_id",
        "timestamp",
        "text",
        "rate",
        "star",
    ]
    user_reviews_selected = user_reviews[user_reviews_features]

    print("Feature selection completed!\n")

    # Save selected features to new CSV files
    user_data_selected_path = os.path.join(output_folder, "user_data_selected.csv")
    user_activities_selected_path = os.path.join(
        output_folder, "user_activities_selected.csv"
    )
    user_reviews_selected_path = os.path.join(
        output_folder, "user_reviews_selected.csv"
    )

    user_data_selected.to_csv(user_data_selected_path, index=False)
    user_activities_selected.to_csv(user_activities_selected_path, index=False)
    user_reviews_selected.to_csv(user_reviews_selected_path, index=False)

    print(
        f"Selected features saved to:\n  {user_data_selected_path}\n  {user_activities_selected_path}\n  {user_reviews_selected_path}"
    )

    print("\nFeature selection process completed.")


if __name__ == "__main__":
    # Update paths as needed
    feature_selection(
        user_data_path="Combined_data/combined_user_data.csv",
        user_activities_path="Combined_data/combined_user_activities.csv",
        user_reviews_path="Combined_data/combined_user_reviews.csv",
        output_folder="experiment_folder/selected_features",
    )
