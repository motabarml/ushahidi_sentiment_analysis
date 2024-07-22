import requests
import json

class SlackNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def send_alert(self, title, monitoring_dict):
        try:
            # Formatting the monitoring dictionary as a string
            monitoring_info = "\n".join(f"{key}: {value}" for key, value in monitoring_dict.items())
            color = "good"
            # check if there is error in monitoring
            if 'error' in ', '.join(monitoring_dict.values()).lower():
                color = "danger"

            # Preparing the payload
            payload = {
                "attachments": [
                    {
                        "color": color,  # Change color to "warning" or "danger" based on the situation
                        "title": title,
                        "text": monitoring_info,
                        "mrkdwn_in": ["text"]
                    }
                ]
            }

            headers = {'Content-Type': 'application/json'}
            # Sending the POST request to the Slack webhook
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers=headers
            )
            
            # Checking for errors
            response.raise_for_status()
        except requests.RequestException as e:
            print(f'An error occurred: {e}')
        except Exception as e:
            print(f'An error occurred: {e}')