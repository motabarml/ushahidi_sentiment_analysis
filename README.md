# Sentiment Analysis with Google Sheets and BigQuery

This repository contains a Python project that performs sentiment analysis on text data fetched from Google BigQuery and updates a Google Sheet with the results. The project integrates various tools and libraries to achieve the following:

- **Fetch Data:** Extract data from a Google BigQuery table.
- **Translate Text:** Translate text data into English using Google Cloud Translation API.
- **Sentiment Analysis:** Apply sentiment analysis using TextBlob, NRC Emotion Lexicon, and NLTK.
- **Update Google Sheets:** Write the processed data back to a Google Sheet.
- **Upload to BigQuery:** Upload the sentiment analysis results to a new BigQuery table.

## Prerequisites

1. **Google Cloud Platform Credentials:**
   - Service account key JSON file for accessing Google Sheets, BigQuery, and Translation APIs.

2. **Python Libraries:**
   - `google-cloud-translate`
   - `pandas`
   - `gspread`
   - `google-auth`
   - `textblob`
   - `nltk`
   - `nrclex`
   - `google-cloud-bigquery`
   - `joblib`
   - `numpy`

   **Install these libraries using pip:**

   ```bash
   pip install google-cloud-translate pandas gspread google-auth textblob nltk nrclex google-cloud-bigquery joblib numpy

## Setup Instructions

**Service Account Credentials:**
Place your service account JSON key file in the ssh/ directory (ensure it's named service_account.json).

**NLTK Data:**
The necessary NLTK data will be automatically downloaded to the ./modules/nltk_data directory.

**Environment Configuration:**
Ensure that the SERVICE_ACCOUNT_FILE variable in the script points to your service account JSON key file.
Replace BQ_PROJECT_ID, BQ_DATASET_ID, and BQ_TABLE_ID with your specific BigQuery project and dataset details.


## Usage
**Run the Script:**
Execute the script to perform sentiment analysis and update Google Sheets and BigQuery.

```bash
python main.py
```

**Google Sheets and BigQuery Configuration:**
Update the sheet_url and sheet_name in the script with your Google Sheets URL and sheet name.
Ensure the BigQuery dataset and table IDs are correctly specified.

**Data Handling:**
The script will fetch data from BigQuery, translate text, perform sentiment analysis, update Google Sheets, and upload the results back to BigQuery.

## Exclusions

**SSH and Virtual Environment (venv) Folders:**
The ssh/ directory and venv/ folder are not included in this repository. Ensure that you have the appropriate credentials and virtual environment setup locally.

## Contributing
Feel free to open issues or submit pull requests if you have suggestions or improvements. Please ensure that any contributions adhere to the project's coding standards and style guidelines.
