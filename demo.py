import pandas as pd
import numpy as np

# Load CSV files (assuming they match the provided documents)
df1 = pd.read_csv("RS_Session_256_AU_95_C.csv")  # EV Sales by State and Vehicle Type (cumulative)
df2 = pd.read_csv("RS_Session_256_AU_2673_3.csv")  # ROs with EV Charging (snapshot)
df3 = pd.read_csv("RS_Session_259_AU_2837_A.csv")  # PCS as of Feb 2024
df4 = pd.read_csv("RS_Session_263_AU_102_A.csv")  # EV Sales by Category (2022, 2023)
df5 = pd.read_csv("RS_Session_265_AU_277_A_to _B_ii.csv")  # Highway Chargers (snapshot)
df6 = pd.read_csv("RS_Session_265_AU_2151_E.csv")  # PCS as of March 2024
df7 = pd.read_csv("RS_Session_266_AS_217_4.csv")  # Energy Requirement (2024-2030)
df8 = pd.read_csv("RS_Session_266_AU_2164_A.csv")  # Total EV Sales and Penetration (cumulative)

# List of DataFrames for standardization
dfs = [df1, df2, df3, df4, df5, df6, df7, df8]

# Standardize column name across all DataFrames
for df in dfs:
    if "State Name" in df.columns:
        df.rename(columns={"State Name": "State/UT"}, inplace=True)

# Define mapping for state name corrections
state_mapping = {
    "Andaman and Nicobar Island": "Andaman and Nicobar Islands",
    "Pondicherry": "Puducherry",
    "Maharastra": "Maharashtra",
    "Andaman and Nicobar": "Andaman and Nicobar Islands"
}

# Apply state name corrections to all DataFrames
for df in dfs:
    if "State/UT" in df.columns:
        df["State/UT"] = df["State/UT"].replace(state_mapping)

# Convert sales columns in df1 to numeric before merging
df1["Two Wheeler"] = pd.to_numeric(df1["Two Wheeler"], errors='coerce')
df1["Three Wheeler"] = pd.to_numeric(df1["Three Wheeler"], errors='coerce')
df1["Four Wheeler"] = pd.to_numeric(df1["Four Wheeler"], errors='coerce')

# Merge state-specific DataFrames on "State/UT"
final_df = df1.merge(df2, on="State/UT", how="outer")\
              .merge(df3, on="State/UT", how="outer")\
              .merge(df5, on="State/UT", how="outer")\
              .merge(df6, on="State/UT", how="outer")\
              .merge(df8, on="State/UT", how="outer")

# Remove "Leh" from the dataset
final_df = final_df[final_df["State/UT"] != "Leh"]

# Rename columns for clarity before imputation
final_df = final_df.rename(columns={
    "Total EV": "Total EVs Sold",
    "% of Share of EV in Total Vehicles Sold": "EV Penetration (%)",
    "Two Wheeler": "Two Wheeler Sales (Cumulative)",
    "Three Wheeler": "Three Wheeler Sales (Cumulative)",
    "Four Wheeler": "Four Wheeler Sales (Cumulative)",
    "No. of PCS as on 31st March 2024": "No. of PCS (Mar 2024)",
    "Number of Charges on Highway": "No. of Highway Chargers",
    "No of RO's where EV Charging Facility available": "No. of ROs with Charging",
    "No. of Operational PCS": "No. of PCS (Feb 2024)"
})

# Impute missing cumulative sales using hybrid strategy
# Step 1: Calculate national totals and known sums
national_total_2w = df4[df4["Category"] == "2 Wheelers"]["2022"].values[0] + df4[df4["Category"] == "2 Wheelers"]["2023"].values[0]  # 631464 + 859376
national_total_3w = df4[df4["Category"] == "3 Wheelers"]["2022"].values[0] + df4[df4["Category"] == "3 Wheelers"]["2023"].values[0]  # 352710 + 582793
national_total_4w = df4[df4["Category"] == "Passenger Vehicles"]["2022"].values[0] + df4[df4["Category"] == "Passenger Vehicles"]["2023"].values[0]  # 38240 + 82105
known_2w = final_df["Two Wheeler Sales (Cumulative)"].sum()  # Sum of known cumulative sales
known_3w = final_df["Three Wheeler Sales (Cumulative)"].sum()
known_4w = final_df["Four Wheeler Sales (Cumulative)"].sum()
remaining_2w = national_total_2w - known_2w
remaining_3w = national_total_3w - known_3w
remaining_4w = national_total_4w - known_4w

# Step 2: Identify missing states and impute
missing_states = final_df["Two Wheeler Sales (Cumulative)"].isna()
total_missing_ev = final_df.loc[missing_states, "Total EVs Sold"].sum()

# Regional adjustment for Southern states
southern_states = ["Andhra Pradesh", "Telangana", "Tamil Nadu", "Karnataka", "Kerala"]
for index, row in final_df[missing_states].iterrows():
    proportion = row["Total EVs Sold"] / total_missing_ev
    if row["State/UT"] in southern_states:
        # South India: ~80% two-wheelers, 15% three-wheelers, 5% four-wheelers
        final_df.loc[index, "Two Wheeler Sales (Cumulative)"] = row["Total EVs Sold"] * 0.8
        final_df.loc[index, "Three Wheeler Sales (Cumulative)"] = row["Total EVs Sold"] * 0.15
        final_df.loc[index, "Four Wheeler Sales (Cumulative)"] = row["Total EVs Sold"] * 0.05
    else:
        # Default for others (e.g., Madhya Pradesh)
        final_df.loc[index, "Two Wheeler Sales (Cumulative)"] = proportion * remaining_2w
        final_df.loc[index, "Three Wheeler Sales (Cumulative)"] = proportion * remaining_3w
        final_df.loc[index, "Four Wheeler Sales (Cumulative)"] = proportion * remaining_4w

# Ensure Dataset 4 sales columns are numeric
df4["2022"] = pd.to_numeric(df4["2022"], errors='coerce')
df4["2023"] = pd.to_numeric(df4["2023"], errors='coerce')

# Extract national EV sales by category from Dataset 4 (2022, 2023)
sales_2022_2w = df4[df4["Category"] == "2 Wheelers"]["2022"].values[0]  # 631464
sales_2023_2w = df4[df4["Category"] == "2 Wheelers"]["2023"].values[0]  # 859376
sales_2022_3w = df4[df4["Category"] == "3 Wheelers"]["2022"].values[0]  # 352710
sales_2023_3w = df4[df4["Category"] == "3 Wheelers"]["2023"].values[0]  # 582793
sales_2022_pv = df4[df4["Category"] == "Passenger Vehicles"]["2022"].values[0]  # 38240
sales_2023_pv = df4[df4["Category"] == "Passenger Vehicles"]["2023"].values[0]  # 82105

# Calculate 2022 and 2023 sales with imputed cumulative values
total_2w = final_df["Two Wheeler Sales (Cumulative)"].sum()
total_3w = final_df["Three Wheeler Sales (Cumulative)"].sum()
total_4w = final_df["Four Wheeler Sales (Cumulative)"].sum()
final_df["Two Wheeler Sales 2022"] = (final_df["Two Wheeler Sales (Cumulative)"] / total_2w) * sales_2022_2w
final_df["Two Wheeler Sales 2023"] = (final_df["Two Wheeler Sales (Cumulative)"] / total_2w) * sales_2023_2w
final_df["Three Wheeler Sales 2022"] = (final_df["Three Wheeler Sales (Cumulative)"] / total_3w) * sales_2022_3w
final_df["Three Wheeler Sales 2023"] = (final_df["Three Wheeler Sales (Cumulative)"] / total_3w) * sales_2023_3w
final_df["Four Wheeler Sales 2022"] = (final_df["Four Wheeler Sales (Cumulative)"] / total_4w) * sales_2022_pv
final_df["Four Wheeler Sales 2023"] = (final_df["Four Wheeler Sales (Cumulative)"] / total_4w) * sales_2023_pv

# Add Energy Requirement columns from Dataset 7 (2024-2030)
energy_data = df7.set_index("Years")["Energy Requirement"].to_dict()
final_df["Energy Requirement 2024-25 (GWh)"] = energy_data["2024-25"]
final_df["Energy Requirement 2025-26 (GWh)"] = energy_data["2025-26"]
final_df["Energy Requirement 2026-27 (GWh)"] = energy_data["2026-27"]
final_df["Energy Requirement 2027-28 (GWh)"] = energy_data["2027-28"]
final_df["Energy Requirement 2028-29 (GWh)"] = energy_data["2028-29"]
final_df["Energy Requirement 2029-30 (GWh)"] = energy_data["2029-30"]

# Convert all numeric columns to numeric type initially
numeric_columns = [
    "Total EVs Sold", "EV Penetration (%)",
    "Two Wheeler Sales (Cumulative)", "Three Wheeler Sales (Cumulative)", "Four Wheeler Sales (Cumulative)",
    "Two Wheeler Sales 2022", "Two Wheeler Sales 2023",
    "Three Wheeler Sales 2022", "Three Wheeler Sales 2023",
    "Four Wheeler Sales 2022", "Four Wheeler Sales 2023",
    "No. of PCS (Feb 2024)", "No. of PCS (Mar 2024)", "No. of Highway Chargers", "No. of ROs with Charging",
    "Energy Requirement 2024-25 (GWh)", "Energy Requirement 2025-26 (GWh)", 
    "Energy Requirement 2026-27 (GWh)", "Energy Requirement 2027-28 (GWh)", 
    "Energy Requirement 2028-29 (GWh)", "Energy Requirement 2029-30 (GWh)"
]
for col in numeric_columns:
    final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

# Round all numeric columns to 0 decimal places
final_df[numeric_columns] = final_df[numeric_columns].round(0)

# Select key columns
key_columns = [
    "State/UT", "Total EVs Sold", "EV Penetration (%)",
    "Two Wheeler Sales (Cumulative)", "Three Wheeler Sales (Cumulative)", "Four Wheeler Sales (Cumulative)",
    "Two Wheeler Sales 2022", "Two Wheeler Sales 2023",
    "Three Wheeler Sales 2022", "Three Wheeler Sales 2023",
    "Four Wheeler Sales 2022", "Four Wheeler Sales 2023",
    "No. of PCS (Feb 2024)", "No. of PCS (Mar 2024)", "No. of Highway Chargers", "No. of ROs with Charging",
    "Energy Requirement 2024-25 (GWh)", "Energy Requirement 2025-26 (GWh)", 
    "Energy Requirement 2026-27 (GWh)", "Energy Requirement 2027-28 (GWh)", 
    "Energy Requirement 2028-29 (GWh)", "Energy Requirement 2029-30 (GWh)"
]
final_df = final_df[key_columns]

# Replace 'NA' strings with NaN (redundant after pd.to_numeric, but kept for safety)
final_df = final_df.replace("NA", pd.NA)

# Save to CSV
final_df.to_csv("india_ev_market_all_years_numeric_imputed_rounded.csv", index=False)

# Display sample output
print("Dataset saved as 'india_ev_market_all_years_numeric_imputed_rounded.csv'")
print("\nFirst 5 rows of the merged dataset:")
print(final_df.head())