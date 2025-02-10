import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split

# ================== CONFIGURATION ===================
INPUT_CSV = "hpfl_merged_data.csv"  # path to your merged CSV
OUTPUT_DIR = "./hpfl_data"  # folder to store train/test JSON
CONFIG_TXT = "config.txt"  # config file for each city

MIN_USERS_PER_CITY = 30
MAX_USERS_PER_CITY = 12700
MIN_RECORDS_PER_USER = 10
TEST_SIZE = 0.2  # 0.2
RANDOM_SEED = 42  # 42

# Maximum code allowed; codes above this get clipped
max_code_allowed = 999

# ====================================================

os.makedirs(f"{OUTPUT_DIR}/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/test", exist_ok=True)

# ----------------- STEP 1: Load CSV -----------------
df = pd.read_csv(INPUT_CSV)
print("Initial df shape:", df.shape)

# Drop unneeded columns but keep gender and age.
df = df.drop(
    columns=[
        "tag",
        "text",
        "review_timestamp",
        "activity_timestamp",
    ],
    errors="ignore",
)
# Select only the relevant columns.
# We drop 'contribution' and 'event_id' (since we will not use them in the knowledge code)
df = df[
    ["user_id", "business_id", "city", "star", "response", "rate", "age", "gender"]
].dropna()
print("After essential columns & dropna:", df.shape)

# Cast columns to appropriate types.
df = df.astype(
    {
        "user_id": int,
        "business_id": int,
        "city": str,
        "star": float,
        "response": float,
        "rate": float,
        "age": str,
        "gender": str,
    }
)


# ---------------- STEP 2: Create Knowledge Codes ----------------
def create_knowledge_codes(row):
    """Discretize selected features into integer codes.
    Uses star, response, age, and gender.
    """
    codes = []

    # Star: Use integer part (bins: 0..4 if star is in a small range, adjust as needed)
    s_bin = int(row["star"] // 1)
    codes.append(s_bin)

    # Response: Use logarithmic scaling (log1p is appropriate when response is between 0 and 1600)
    r_bin = int(np.log1p(row["response"]))
    codes.append(r_bin)

    # Age: Encode as categorical.
    # Assuming age is given like "90后", we remove the trailing character and convert to integer.
    age_val = row["age"].strip()
    if age_val.endswith("后"):
        try:
            age_code = int(age_val[:-1])
        except:
            age_code = 0
    else:
        try:
            age_code = int(age_val)
        except:
            age_code = 0
    codes.append(age_code)

    # Gender: Replace empty/missing values with "0" for unknown.
    # Then map: "0" -> 1, "male"/"m" -> 2, "female"/"f" -> 3.
    gender_val = (
        row["gender"].strip().lower() if isinstance(row["gender"], str) else "0"
    )
    if gender_val == "":
        gender_val = "0"
    gender_mapping = {"0": 1, "male": 2, "m": 2, "female": 3, "f": 3}
    g_code = gender_mapping.get(gender_val, 1)
    codes.append(g_code)

    # Remove duplicates and return
    return list(set(codes))


df["knowledge_code"] = df.apply(create_knowledge_codes, axis=1)


# ---------------- STEP 3: Clip codes to max_code_allowed --------
def clip_codes(codes, max_code):
    clipped = []
    for c in codes:
        if c < 0:
            c = 0
        if c > max_code:
            c = max_code
        clipped.append(c)
    return clipped


df["knowledge_code"] = df["knowledge_code"].apply(
    lambda codes: clip_codes(codes, max_code_allowed)
)

# knowledge_n is now set to max_code_allowed + 1
knowledge_n = max_code_allowed + 1
print("Final knowledge_n =", knowledge_n)

# ---------------- STEP 4: Filter Out Small Cities ---------------
city_stats = (
    df.groupby("city")
    .agg(num_users=("user_id", "nunique"), total_records=("user_id", "count"))
    .reset_index()
)
valid_cities = city_stats[
    (city_stats["num_users"] >= MIN_USERS_PER_CITY)
    & (city_stats["num_users"] <= MAX_USERS_PER_CITY)
    & (city_stats["total_records"] >= MIN_USERS_PER_CITY * MIN_RECORDS_PER_USER)
]["city"].unique()

df = df[df["city"].isin(valid_cities)].copy()
print("Number of valid cities:", len(valid_cities))

valid_cities_list = sorted(valid_cities)
city2id = {city_name: i for i, city_name in enumerate(valid_cities_list)}

# ---------------- STEP 5: Create config.txt & JSON --------------
config_lines = []

for city_name in valid_cities_list:
    city_id = city2id[city_name]

    # Subset for this city
    city_df = df[df["city"] == city_name].copy()

    # Build local user & biz maps
    unique_users = sorted(city_df["user_id"].unique())
    user2idx = {u: i for i, u in enumerate(unique_users)}
    user_n = len(user2idx)

    unique_biz = sorted(city_df["business_id"].unique())
    biz2idx = {b: i for i, b in enumerate(unique_biz)}
    biz_n = len(biz2idx)

    # Convert each row into a record with local user ID and knowledge_code.
    records = []
    for _, row in city_df.iterrows():
        local_uid = user2idx[row["user_id"]]
        local_biz = biz2idx[row["business_id"]]

        record = {
            "user_id": local_uid,
            "exer_id": local_biz,
            "score": float(row["rate"]) / 50.0,  # Normalize to 0-1 range
            "knowledge_code": row["knowledge_code"],  # already processed and clipped
        }
        records.append(record)

    # Split into training and testing sets
    train_records, test_records = train_test_split(
        records, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # Save JSON files
    train_path = f"{OUTPUT_DIR}/train/{city_id}.json"
    test_path = f"{OUTPUT_DIR}/test/{city_id}.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_records, f, ensure_ascii=False)

    # Config line format: city_id user_n biz_n knowledge_n
    config_lines.append(f"{city_id} {user_n} {biz_n} {knowledge_n}")

# Write the config file
with open(CONFIG_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(config_lines))
    f.write("\n")

print("Done! Wrote JSON to", OUTPUT_DIR, "and config to", CONFIG_TXT)
print("You can now train with city-based IDs and knowledge_dim =", knowledge_n)
