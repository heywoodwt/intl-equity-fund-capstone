import pdfplumber
import re
import csv

# --- Configuration ---
pdf_filename = "annual_custody_statement_2022.pdf"
csv_filename = "purchase_and_sale_transactions_full_desc_2022.csv"

# --- Regular Expressions ---
# Matches the start of a Purchase/Sale transaction line
transaction_start_pattern = re.compile(r"^(\d{2}/\d{2}/\d{2})\s+(Purchase|Sale)\s+(.*?)\s+(-?[\d,]+\.\d{2})$")

# Matches any line starting with a date (useful to prevent appending other transaction types like "Dividends" to our descriptions)
date_pattern = re.compile(r"^\d{2}/\d{2}/\d{2}")

# Words or phrases that indicate a page header/footer that we should ignore
ignore_phrases = [
    "Date Posted", "Activity", "Description", "Principal Cash", 
    "ACCOUNT NUMBER", "Page", "Usbank",
    "THIS PAGE WAS INTENTIONALLY LEFT BLANK"
]

extracted_data = []
current_transaction = None

print(f"Opening '{pdf_filename}'...")

try:
    with pdfplumber.open(pdf_filename) as pdf:
        total_pages = len(pdf.pages)
        
        for i, page in enumerate(pdf.pages):
            if (i + 1) % 10 == 0 or (i + 1) == total_pages:
                print(f"Scanning page {i + 1} of {total_pages}...")
                
            text = page.extract_text()
            if not text:
                continue
                
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip known page headers and footers
                if any(phrase.lower() in line.lower() for phrase in ignore_phrases):
                    continue
                    
                # Check if the line is the START of a new Purchase/Sale transaction
                match = transaction_start_pattern.match(line)
                
                if match:
                    # If we were already tracking a transaction, save it before starting this new one
                    if current_transaction:
                        extracted_data.append([
                            current_transaction['date'], 
                            current_transaction['activity'], 
                            current_transaction['desc'], 
                            current_transaction['amount']
                        ])
                        
                    # Start tracking the new transaction
                    current_transaction = {
                        'date': match.group(1),
                        'activity': match.group(2),
                        'desc': match.group(3).strip(),
                        'amount': match.group(4)
                    }
                
                # If it's NOT a new Purchase/Sale transaction, it might be spillover text
                elif current_transaction:
                    # Check that it doesn't start with a date (which would mean it's a different transaction type, like a Fee or Dividend)
                    if not date_pattern.match(line):
                        # Append this line to the current transaction's description
                        current_transaction['desc'] += " " + line

        # Don't forget to save the very last transaction in the document!
        if current_transaction:
            extracted_data.append([
                current_transaction['date'], 
                current_transaction['activity'], 
                current_transaction['desc'], 
                current_transaction['amount']
            ])

    print(f"\nSuccess! Found {len(extracted_data)} Purchase/Sale transactions with full descriptions.")

    # --- Save to CSV ---
    print(f"Saving data to '{csv_filename}'...")
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Date Posted", "Activity", "Description", "Principal Cash"])
        writer.writerows(extracted_data)
        
    print("Done! Your CSV is ready.")

except FileNotFoundError:
    print(f"Error: Could not find the file '{pdf_filename}'. Please make sure it is in the same folder as this script.")
except Exception as e:
    print(f"An error occurred: {e}")