import re
import json
import requests

def replace_hex_to_ascii(match):
    hex_value = match.group(1)
    try:
        ascii_char = chr(int(hex_value, 16))
        return ascii_char
    except ValueError:
        return match.group(0)  # Return the original string if conversion fails

def replace_quotes(text):
    return text.replace('&quot;', '"')

def replace_html_entities(text):
    # Find all occurrences of '&#xHH;' and replace with corresponding ASCII character
    pattern = r'&#x([0-9a-fA-F]{2});'
    replaced_text = re.sub(pattern, replace_hex_to_ascii, text)
    return replaced_text

def create_credentials_cookies(cookies, bl_credentials):
    cookies.update(bl_credentials)
    

def requests_request(url, cookies, headers, method='GET', params=None, json=None):
    """"""
    response = requests.request(
        url=url,
        cookies=cookies,
        headers=headers,
        method=method,
        params=params,
        json=json
    )
    
    return response