# This file is for manually inputting links. It is used for finding the bias rating of each news outlet and for general testing
# without needing to run the FastAPI frontend.
# To run this file alone, you can click the |> button in the top right corner of the code editor.

import pandas as pd
from openpyxl import load_workbook
import os
from article_utils import fetch_article, is_restricted_url, restricted_outlets
from classify_articles import classify_bias

# Initializing the Excel file and sheet names
# This is where the data will be saved.
excel_file = "ExtractedData.xlsx"
full_sheet = 'Sheet1'
summary_sheet = 'Sheet2'

# You can put a list of articles here and run this file individually to get their rating. Does not require front end running       
article_urls = [ 
    'https://ottawa.citynews.ca/2025/03/29/carney-campaigning-in-his-ottawa-riding-today-poilievre-in-winnipeg/', 
    'https://vocm.com/2025/04/05/266777/',
    'https://www.cbc.ca/news/politics/singh-promises-more-doctors-carney-supports-the-trades-poilievre-vows-to-cut-red-tape-1.7502992',
    'https://nationalpost.com/news/politics/federal_election/less-red-tape-more-trades-workers-and-doctors-for-all-on-campaign-day-14',
    'https://www.theglobeandmail.com/politics/article-singh-promises-to-add-up-to-7500-family-doctors-in-the-next-five-years/',
    'https://globalnews.ca/news/11116941/carney-singh-pledge-support-for-cbc-radio-canada-amid-u-s-threats/',
]

# Lists to hold data to be written into Excel
new_full_data = []
new_summary_data = []

# Loop through each URL and process it
for url in article_urls:
    # Check if the URL is from a restricted outlet and skip it if so
    if is_restricted_url(url):
        print(f"Access restricted: {url}")
        continue

    title, content = fetch_article(url)
    if title and content:
        raw_response = classify_bias(content) or ""  
        bias = raw_response.strip().capitalize() if raw_response.strip() else "Unknown"

        # Print the results for each article
        print(f"\nRaw model output: '{raw_response}'")
        print(f"Final bias classification: {bias}\n")

        # Collect full and summary data
        new_full_data.append({
            'Title': title,
            'Content': content,
            'URL': url,
            'Bias': bias,
            'RawOutput': raw_response
        })

        new_summary_data.append({
            'Title': title,
            'Bias': bias,
            'RawOutput': raw_response
        })

df_full = pd.DataFrame(new_full_data)
df_summary = pd.DataFrame(new_summary_data)

if os.path.exists(excel_file):
    book = load_workbook(excel_file)

    # Append data to existing sheets or create new ones if they don't exist
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        if full_sheet in book.sheetnames:
            startrow_full = book[full_sheet].max_row
            df_full.to_excel(writer, sheet_name=full_sheet, index=False, header=startrow_full == 1, startrow=startrow_full)
        else:
            df_full.to_excel(writer, sheet_name=full_sheet, index=False)

        if summary_sheet in book.sheetnames:
            startrow_summary = book[summary_sheet].max_row
            df_summary.to_excel(writer, sheet_name=summary_sheet, index=False, header=startrow_summary == 1, startrow=startrow_summary)
        else:
            df_summary.to_excel(writer, sheet_name=summary_sheet, index=False)

else:
    # If the file doesn't exist, create it and write the data
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df_full.to_excel(writer, sheet_name=full_sheet, index=False)
        df_summary.to_excel(writer, sheet_name=summary_sheet, index=False)

# Save the new data to the Excel file
print(f"\nSaved {len(new_full_data)} articles to '{excel_file}' in sheets '{full_sheet}' and '{summary_sheet}'.")
