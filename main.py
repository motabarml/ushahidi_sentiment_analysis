from google.cloud import translate_v2 as translate
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
from nltk.tokenize import word_tokenize
from nrclex import NRCLex
import os
from google.cloud import bigquery
from joblib import Parallel, delayed
import numpy as np

from modules.utils._credentials import LocalCredentials
import yaml

# Download necessary NLTK resources
nltk_data_dir = './modules/nltk_data'

if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

download('vader_lexicon', download_dir=nltk_data_dir)
download('punkt', download_dir=nltk_data_dir)

# Define the scope for Google Sheets and Google Drive APIs
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
#TODO:
SERVICE_ACCOUNT_FILE = 'ssh/service_account.json'  # Replace with your service account key file path

def authorize_google_sheets():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client = gspread.authorize(creds)
    return client

def get_google_sheet(sheet_url, sheet_name):
    client = authorize_google_sheets()
    sheet = client.open_by_url(sheet_url).worksheet(sheet_name)
    
    # Fetch data and headers
    records = sheet.get_all_values()
    
    if len(records) == 0:
        raise ValueError("The sheet is empty.")
    
    headers = records[0]
    data = records[1:]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)
    
    return df, sheet

def translate_batch(text_batch):
    if not text_batch:
        return []
    
    try:
        client = translate.Client.from_service_account_json(SERVICE_ACCOUNT_FILE)
        results = client.translate(text_batch, target_language='en')
        return [result['translatedText'] for result in results]
    except Exception as e:
        print(f"Batch translation error: {e}")
    return text_batch

# BigQuery configuration
# TODO:
BQ_PROJECT_ID = ''  # Replace with your GCP project ID
BQ_DATASET_ID = ''  # Replace with your BigQuery dataset ID
BQ_TABLE_ID = '' # Replace with your BigQuery Table ID
BQ_NEW_TABLE_ID =  '' # Replace with your BigQuery table ID

def fetch_from_bigquery(query):
    bq_client = bigquery.Client.from_service_account_json(SERVICE_ACCOUNT_FILE, project=BQ_PROJECT_ID)
    query_job = bq_client.query(query)
    results = query_job.result()
    df = results.to_dataframe()
    return df

# Define query to fetch data from BigQuery
query = f"""
SELECT * 
FROM `{BQ_TABLE_ID}`
"""

# Fetch data from BigQuery
bq_data = fetch_from_bigquery(query)

# Save Data locally (Optional)
# csv_file_path = 'bq_data.csv'
# bq_data.to_csv(csv_file_path, index=False)
# print(f"Data saved to {csv_file_path}")

#TODO:
# Replace the Google Sheets URL and sheet name with your own
sheet_url = ''
sheet_name = ''

# Update Google Sheet with BigQuery data
def update_google_sheet(sheet_url, sheet_name, df):
    client = authorize_google_sheets()
    sheet = client.open_by_url(sheet_url).worksheet(sheet_name)
    
    # Convert DataFrame to list of lists
    df_list = [df.columns.tolist()] + df.values.tolist()
    
    # Ensure no ndarray objects are included
    df_list = [[str(cell) if isinstance(cell, (int, float)) else str(cell) for cell in row] for row in df_list]
    
    # Clear the sheet before updating
    sheet.clear()
    
    # Update the sheet with new data
    sheet.update('A1', df_list)

update_google_sheet(sheet_url, sheet_name, bq_data)

# Read the updated data from Google Sheets
ushahidi, sheet = get_google_sheet(sheet_url, sheet_name)


# Define a function to split a list into chunks
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

# Define the function to translate text in parallel
def translate_text_batch(text_batch):
    client = translate.Client.from_service_account_json(SERVICE_ACCOUNT_FILE)
    translated_texts = []
    max_chunk_size = 128  # API limit for text segments per request
    
    for chunk in chunk_list(text_batch, max_chunk_size):
        results = client.translate(chunk, target_language='en')
        translated_texts.extend(result['translatedText'] for result in results)
    
    return translated_texts

# Number of parallel jobs
num_jobs = 6

# Convert memmap arrays to lists
title_list = ushahidi['title'].dropna().tolist()
content_list = ushahidi['content'].dropna().tolist()

# Split data into batches for parallel processing
title_batches = np.array_split(title_list, num_jobs)
content_batches = np.array_split(content_list, num_jobs)

# Translate 'title' and 'content' columns in parallel
print('Translating the title and content to English...')
translated_titles = Parallel(n_jobs=num_jobs)(
    delayed(translate_text_batch)(batch.tolist()) for batch in title_batches
)
translated_contents = Parallel(n_jobs=num_jobs)(
    delayed(translate_text_batch)(batch.tolist()) for batch in content_batches
)

# Flatten the results
ushahidi['title_translated'] = [item for sublist in translated_titles for item in sublist]
ushahidi['content_translated'] = [item for sublist in translated_contents for item in sublist]

# Initialize the Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Define a function to calculate TextBlob sentiment
def get_textblob_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Apply the TextBlob sentiment function
print('Applying TextBlob Sentiment...')
ushahidi['sentiment'] = ushahidi['content_translated'].apply(get_textblob_sentiment)

# Replace NA values with 0 for numeric columns
ushahidi.fillna(0, inplace=True)

# Calculate content length (number of characters)
ushahidi['content_length'] = ushahidi['content_translated'].apply(len)

# Calculate the number of words in content using NLTK's word_tokenize
ushahidi['content_words'] = ushahidi['content_translated'].apply(lambda x: len(word_tokenize(x)))

# Define a function to calculate NRC sentiment using NRCLex
def get_nrc_sentiment(text):
    text_object = NRCLex(text)
    return text_object.raw_emotion_scores

# Apply the NRC sentiment function to the 'content_translated' column
sentiment_scores = ushahidi['content_translated'].apply(get_nrc_sentiment)

# Convert the sentiment scores to a DataFrame
sentiment_df = pd.DataFrame(list(sentiment_scores))

# Concatenate the sentiment scores DataFrame with the original DataFrame
ushahidi = pd.concat([ushahidi, sentiment_df], axis=1)

# Replace NA values with 0 for numeric columns
ushahidi.fillna(0, inplace=True)

print(ushahidi.head())

credentials_instance = LocalCredentials()


client = credentials_instance.bigquery_client()
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE"  # Change to "WRITE_APPEND" if you want to append data
)

print('Uploading Sentiment Analysis Data now...')
job = client.load_table_from_dataframe(
    ushahidi, f"{BQ_DATASET_ID}.{BQ_NEW_TABLE_ID}", job_config=job_config
)
job.result()  # Wait for the job to complete
print('Uploaded Sentiment Analysis Data to BigQuery successfully.')

