#github would let me push the data because it was larger than 100GB
import pandas as pd

import pandas as pd

def clean_and_save(file_name, rows_to_drop=2000):
    # 1. Load the file
    df = pd.read_csv(file_name)
    
    # 2. Check if we have enough rows to drop
    if len(df) > rows_to_drop:
        # Sample the indices and drop them
        indices = df.sample(n=rows_to_drop).index
        df_cleaned = df.drop(indices)
        
        # 3. Save it back to the same filename
        df_cleaned.to_csv(file_name, index=False)
        print(f"Processed {file_name}: Dropped {rows_to_drop} rows. New count: {len(df_cleaned)}")

files = ['LA_listings_Q1.csv', 'LA_listings_Q2.csv']

# Run the function for each
for file in files:
    clean_and_save(file)