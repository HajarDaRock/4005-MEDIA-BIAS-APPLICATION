import requests
from requests.exceptions import Timeout, HTTPError
from bs4 import BeautifulSoup

# List of news outlets that are restricted from scraping
restricted_outlets = [
    'thehill.com',
    'ctvnews.ca',
    'ipsos.com',
    'nytimes.com'
]

def is_restricted_url(url):
    """
    Checks whether a given URL belongs to a restricted news outlet.

    Parameters:
        url (str): The URL to be checked.

    Returns:
        bool: True if the URL is from a restricted outlet, False otherwise.
    """
    return any(outlet in url for outlet in restricted_outlets)

def fetch_article(url):
    """
    Fetches and parses the article content from the given URL.

    Parameters:
        url (str): The web address of the article to fetch.

    Returns:
        tuple: A tuple containing the title and the full article text.
               Returns (None, None) if an error occurs or the request fails.
    """
    # Custom user-agent to reduce the risk of being blocked by some websites
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/91.0.4472.124 Safari/537.36'
        )
    }

    try:
        # Send GET request with custom headers and a timeout
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for HTTP error codes

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the title from the first <h1> tag
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else 'No Title Found'

        # Extract all text inside <p> tags and join into a single string
        paragraphs = soup.find_all('p')
        content = ' '.join(p.get_text(strip=True) for p in paragraphs)

        return title, content

    except Timeout:
        # Handle case where the request exceeds the timeout limit
        print(f"Request to {url} timed out after 10 seconds.")
        return None, None

    except HTTPError as http_err:
        # Handle specific HTTP errors (e.g., 403 Forbidden)
        if response.status_code == 403:
            print(f"Access to {url} is forbidden. Unable to access the article.")
        else:
            print(f"HTTP error occurred: {http_err}")
        return None, None

    except Exception as e:
        # Handle any other exceptions
        print(f"An error occurred while fetching the article: {e}")
        return None, None
