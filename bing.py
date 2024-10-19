import os
import time
import logging
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def searchbing(query, page_num):
    """
    Calls the Bing Search API and returns results based on the query and page number.

    Args:
        query (str): The search query string.
        page_num (int): The number of result pages to return.
        
    Returns:
        dict: A dictionary in JSON format containing the search results.
    """
    
    subscription_key = os.getenv('BING_API_KEY', 'your_default_key_here')
    if not subscription_key:
        logging.error("Bing API key is missing. Set the BING_API_KEY environment variable.")
        return {}

    url = "https://api.bing.microsoft.com/v7.0/search"
    market = 'en-US'
    answer_count = 2
    
    params = {
        'q': query,
        'mkt': market,
        'answerCount': answer_count,
        'count': page_num,
        'responseFilter': ['Webpages']
    }
    
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key
    }


    retry_interval_exp = 0
    max_retries = 3

    while retry_interval_exp <= max_retries:
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            logging.info("Bing API request successful.")
            return response.json()

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            logging.error(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logging.error(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logging.error(f"Request error occurred: {req_err}")
        
        sleep_time = max(2, 0.5 * (2 ** retry_interval_exp))
        logging.warning(f"Retrying in {sleep_time} seconds... (attempt {retry_interval_exp + 1}/{max_retries + 1})")
        time.sleep(sleep_time)
        retry_interval_exp += 1
    
    logging.error("Maximum retry limit reached. Failed to retrieve Bing search results.")
    return {}

if __name__ == "__main__":
    query = "Python programming"
    page_num = 1
    result = searchbing(query, page_num)
    pprint(result)
