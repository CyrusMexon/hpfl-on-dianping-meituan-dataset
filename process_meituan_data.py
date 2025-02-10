import os
import pandas as pd


# Function to read files from a folder and combine them into a single DataFrame
def read_data_from_folder(folder_path):
    """
    Reads all .xlsx and .txt files from the given folder and combines them into a single DataFrame.

    Args:
        folder_path (str): Path to the folder containing data files.

    Returns:
        pd.DataFrame: Combined DataFrame of all files in the folder.
    """
    data_frames = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Read .xlsx files
        if file_name.endswith(".xlsx"):
            df = pd.read_excel(file_path)
            data_frames.append(df)

        # Read .txt files (assuming tab-separated or comma-separated)
        elif file_name.endswith(".txt"):
            try:
                df = pd.read_csv(file_path, sep="\t")  # Adjust separator if needed
            except:
                df = pd.read_csv(file_path)  # Try comma-separated if tab fails
            data_frames.append(df)

    # Combine all DataFrames in the folder
    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        return combined_df
    else:
        print(f"No valid data files found in {folder_path}")
        return pd.DataFrame()  # Return an empty DataFrame if no files found


# Main function to process the Meituan data
def process_meituan_data(base_path):
    """
    Processes Meituan data from multiple folders (用户数据, 用户活动, 用户评论).

    Args:
        base_path (str): Base directory containing the data folders.

    Returns:
        None
    """
    # Define folder paths
    user_data_folder = os.path.join(base_path, "用户数据")
    user_activities_folder = os.path.join(base_path, "用户活动")
    user_reviews_folder = os.path.join(base_path, "用户评论")

    # Read data from folders
    print("Loading 用户数据...")
    user_data = read_data_from_folder(user_data_folder)
    print("Loading 用户活动...")
    user_activities = read_data_from_folder(user_activities_folder)
    print("Loading 用户评论...")
    user_reviews = read_data_from_folder(user_reviews_folder)

    # Save combined data to CSV for later use
    if not user_data.empty:
        user_data.to_csv("combined_user_data.csv", index=False)
        print("Saved combined 用户数据 to combined_user_data.csv")
    else:
        print("No valid data found for 用户数据.")

    if not user_activities.empty:
        user_activities.to_csv("combined_user_activities.csv", index=False)
        print("Saved combined 用户活动 to combined_user_activities.csv")
    else:
        print("No valid data found for 用户活动.")

    if not user_reviews.empty:
        user_reviews.to_csv("combined_user_reviews.csv", index=False)
        print("Saved combined 用户评论 to combined_user_reviews.csv")
    else:
        print("No valid data found for 用户评论.")

    print("Processing completed!")


if __name__ == "__main__":
    # Update this path to point to your Meituan data directory
    base_path = "from_company"

    if not os.path.exists(base_path):
        print(
            f"Base path '{base_path}' does not exist. Please check the directory path."
        )
    else:
        process_meituan_data(base_path)
