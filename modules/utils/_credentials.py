from google.oauth2 import service_account
import json
import os
import io
import base64
# from google.cloud import storage
from google.cloud import bigquery
from modules.utils._globals import run_locally
import yaml


class LocalCredentials:
    
    def __init__(self) -> None:
        # self.file_path = file_path
        self.run_locally = run_locally
        if not self.run_locally:
            pass
            # storage_client = storage.Client()
            # self.storage_client = storage_client
            
    
            

    def gcp_credentials(self, encoded=False):
        """gcp_credentials

        Returns:
                instance of google.oauth2.service_account.Credentials
        """
        print("--- getting gcp_credentials() ---")
        service_account_string = self.read_service_account_credentials_file()
        if encoded:
            service_account_string = base64.b64decode(service_account_string).decode()
        try:
            credentials = service_account.Credentials.from_service_account_info(
                json.loads(service_account_string, strict=False),
                scopes=[
                    "https://www.googleapis.com/auth/bigquery"
                ],
            )
            return credentials
        except Exception as e:
            print(f"error in gcp_credentials: {str(e)}")
            return None

    def bigquery_client(self, encoded=False):
        """returns client of class bigquery
        Args:
            None
        Returns:
            bq_client: object of class bigquery.Client()
        """
        print(" --- creating bigquery_client() ---")
        try:
            _credentials = self.gcp_credentials(encoded)
            bq_client = bigquery.Client(
                credentials=_credentials,
                project=_credentials.project_id,
            )
            return bq_client
        except Exception as e:
            print(f"error in bigquery_client() \nError: {str(e)}")
            return None

    def read_service_account_credentials_file(self):
        """read data of the service account file as return as string
        Args:
            None

        Returns:
            service_account_string: string
        """

        if self.run_locally:
            # read local files
            service_account_string = self.read_local_credential_file(
                "ssh\service_account.json"
                # "ssh/service_account.json"
            )
        else:
            # read from cloud storage
            # we don't have any storage for our function, so we will create file object
            # in-memory
            service_account_string = self.read_environment_var(
                "PATH_CREDS_SERVICE_ACCOUNT"
            )
            # get credentials
        return service_account_string

    def gmail_credentials(self, _type: str):
        """return required
        Args:
            _type: (str) type of the credentials required [personal, ml]

        Returns:
            gmail_creds: (list) list with username and password
        """
        print("--- reading Gmail Credentials ---")
        if self.run_locally:
            gmail_creds = self.read_local_credential_file("ssh/gmail_creds.json")
        else:
            print("reading blob")
            gmail_creds = self.download_blob_as_bytes("GMAIL_PASS")
        gmail_creds = json.loads(gmail_creds)[_type]
        return gmail_creds
    
    def slack_webhook(self):
        """reading slack webhook to send alerts
        Args:
            None
        Returns:
            webhook: (str) slack channel webhook
        """
        print("--- reading slack_webhook() ---")
        if self.run_locally:
            webhook = self.read_local_credential_file("ssh/slack_webhook")
            return webhook
        else:
            # print('reading blob')
            webhook = self.read_environment_var("SLACK_WEBHOOK")
            return webhook

    def read_environment_var(self, path):
        """"""
        value = os.environ.get(path, f"path to {path} pass not set")
        return value
    
    def read_local_credential_file(self, file_path):
        """read local file

        Args:
            file_path: string path of file

        Returns:
            file_content: file content as string
        """
        f = open(file_path, "r")
        file_content = f.read()
        f.close()
        return file_content
    
    