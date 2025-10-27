#Scrapped
"""import requests
from bs4 import BeautifulSoup

def fetch_ctv_article(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Locate the article body
        article_body = soup.find('article', class_='b-article-body')
        if not article_body:
            print("Article body not found.")
            return None, None

        # Extract the title
        title_tag = article_body.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else 'No Title Found'

        # Extract paragraphs
        paragraphs = article_body.find_all('p', class_='c-paragraph')
        content = ' '.join(p.get_text(strip=True) for p in paragraphs)

        return title, content

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# Example usage
url = 'https://www.ctvnews.ca/montreal/article/welcome-sign-with-image-of-woman-wearing-hijab-officially-removed-by-montreal-city-hall/'
title, content = fetch_ctv_article(url)
print(f"Title: {title}\nContent: {content}")
"""