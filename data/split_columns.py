import pandas as pd
import re

# 1. Load the original CSV file
df = pd.read_csv('purchase_and_sale_transactions_full_desc_2022.csv')

# 2. Define a function to extract the relevant parts using Regex
def extract_fields(desc):
    # Initialize an empty dictionary for our new columns
    data = {
        'Quantity': '',
        'Unit Type': '',
        'Asset Name': '',
        'Trade Date': '',
        'Broker': '',
        'Price Per Unit': ''
    }
    
    # Clean up garbage strings (e.g., long numbers like 111111111)
    desc = re.sub(r'\s*\d{20,}\s*', ' ', str(desc))
    desc = re.sub(r'TRANSACTION DETAIL \(continued\)', '', desc, flags=re.IGNORECASE).strip()
    
    # Extract Quantity and Unit Type (e.g., "Purchased 300 Shares")
    qty_match = re.search(r'(?:Purchased|Sold) ([\d,.]+) (Shares|Units)', desc, re.IGNORECASE)
    if qty_match:
        data['Quantity'] = qty_match.group(1)
        data['Unit Type'] = qty_match.group(2)
        
    # Extract Asset Name (Text between "Of" and "Trade Date")
    asset_match = re.search(r'Of(?:\s+-?[\d,.]+\s+[\d,.]+)?\s+(.*?)\s+Trade Date', desc, re.IGNORECASE)
    if asset_match:
        data['Asset Name'] = asset_match.group(1).strip()

    # Extract Trade Date
    td_match = re.search(r'Trade Date (\d{1,2}/\d{1,2}/\d{2,4})', desc, re.IGNORECASE)
    if td_match:
        data['Trade Date'] = td_match.group(1)
        
    # Extract Broker (Text between "Through" and "Paid" or "Purchased")
    broker_match = re.search(r'(?:Purchased|Sold) Through (.*?)(?:\s+Paid|\s+Purchased|\s+Sold|\s+\d+ Shares|\s+\d+ Units|$)', desc, re.IGNORECASE)
    if broker_match:
        data['Broker'] = broker_match.group(1).strip()
        
    # Extract Price Per Unit (Matches numbers and currency at the end of the text)
    price_matches = re.findall(r'At ([\d.,]+\s*(?:USD|EUR|CHF|GBP|SEK|NOK|DKK|JPY)?)', desc, re.IGNORECASE)
    if price_matches:
        data['Price Per Unit'] = price_matches[-1].strip()
        
    return pd.Series(data)

# 3. Apply the extraction function to the 'Description' column
split_cols = df['Description'].apply(extract_fields)

# 4. Combine the original data (dropping the bulky description) with our new clean columns
final_df = pd.concat([df.drop('Description', axis=1), split_cols], axis=1)

# Reorder columns to make it look nice
final_df = final_df[['Date Posted', 'Activity', 'Quantity', 'Unit Type', 'Asset Name', 'Trade Date', 'Broker', 'Price Per Unit', 'Principal Cash']]

# 5. Save the clean data to a new CSV
final_df.to_csv('split_transactions_2022.csv', index=False)
print("Processing complete! Saved to split_transactions_2022.csv")