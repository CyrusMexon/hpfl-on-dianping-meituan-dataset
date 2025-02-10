import pandas as pd
import numpy as np


def expand_rows_with_multiple_ids(df, col="business_id"):
    """
    Splits rows where 'col' contains multiple comma-separated IDs
    into multiple rows, each with one ID.

    Example:
        If df[col] has "123,456", we create two rows: one with 123, another with 456.
    """
    new_rows = []
    for _, row in df.iterrows():
        # If this row has multiple IDs, split them. Otherwise just one ID.
        business_ids = str(row[col]).split(",")
        for b in business_ids:
            b = b.strip()  # remove whitespace
            new_row = row.copy()
            new_row[col] = b  # place the single ID
            new_rows.append(new_row)
    expanded_df = pd.DataFrame(new_rows)

    # Convert the col to numeric if possible (some might fail, e.g. non-numeric strings).
    expanded_df[col] = pd.to_numeric(expanded_df[col], errors="coerce")

    # Drop rows where conversion failed (NaN).
    expanded_df.dropna(subset=[col], inplace=True)

    # Convert to int64 finally.
    expanded_df[col] = expanded_df[col].astype("int64")

    return expanded_df


def main():
    # 1. Load the CSVs
    user_data = pd.read_csv("selected_features/user_data_selected.csv")
    user_activities = pd.read_csv("selected_features/user_activities_selected.csv")
    user_reviews = pd.read_csv("selected_features/user_reviews_selected.csv")

    # 2. Rename timestamp columns
    user_activities.rename(columns={"timestamp": "activity_timestamp"}, inplace=True)
    user_reviews.rename(columns={"timestamp": "review_timestamp"}, inplace=True)

    # 3. Expand user_activities rows that have multiple business_id entries
    print("Expanding multiple business IDs in user_activities...")
    user_activities_expanded = expand_rows_with_multiple_ids(
        user_activities, col="business_id"
    )
    print(f"user_activities_expanded now has {len(user_activities_expanded)} rows.")

    # 4. First merge user_data (which has city) with user_reviews (which has rate)
    #    on `user_id` only, so each row with a `rate` can inherit the user's `city`.
    print("Merging user_data with user_reviews on `user_id` (outer join)...")
    df_ur = pd.merge(user_data, user_reviews, on="user_id", how="outer")

    # 5. Merge the result with user_activities_expanded on [user_id, business_id]
    print(
        "Merging df_ur with user_activities_expanded on [user_id, business_id] (outer join)..."
    )
    df_uar = pd.merge(
        df_ur, user_activities_expanded, on=["user_id", "business_id"], how="outer"
    )

    # 6. Handle missing values more selectively.
    #    Let's define which columns are text vs numeric for a more refined fill.
    #    Adjust these lists as needed based on your actual columns.
    text_cols = [
        "age",
        "city",
        "tag",
        "activity_timestamp",
        "review_timestamp",
        "text",
        # add more if you have other string columns
    ]
    numeric_cols = [
        "contribution",
        "rate",
        "star",
        "response",
        "is_chosen",
        # add more if you have other numeric columns
    ]

    # Fill text columns with "" where missing
    for col in text_cols:
        if col in df_uar.columns:
            df_uar[col] = df_uar[col].fillna("").astype(str)

    # Fill numeric columns with 0 where missing
    for col in numeric_cols:
        if col in df_uar.columns:
            df_uar[col] = pd.to_numeric(df_uar[col], errors="coerce").fillna(0)

    # 7. Save final merged dataset
    df_uar.to_csv("hpfl_merged_data.csv", index=False)
    print("Merged data saved to hpfl_merged_data.csv")

    # 8. Check final dtypes
    print("\nFinal column dtypes:")
    print(df_uar.dtypes)

    # 9. Inspect how many rows actually have non-empty city and non-zero rate
    #    Just to confirm that city is broadcast to rating rows.
    has_city_and_rate = df_uar[
        (df_uar["city"] != "") & (df_uar["city"] != "0") & (df_uar["rate"] != 0)
    ]
    print(f"\nRows with both city != '' and rate != 0: {len(has_city_and_rate)}")


if __name__ == "__main__":
    main()
