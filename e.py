import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_amazon_reviews(product_url, total_reviews=10000):
    reviews = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    for page_number in range(1, (total_reviews // 10) + 1):
        url = f'{product_url}&pageNumber={page_number}'
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            review_elements = soup.find_all('div', class_='a-section review aok-relative')
            for review_element in review_elements:
                review_text = review_element.find('span', class_='a-size-base review-text review-text-content').text.strip()
                review_score = int(review_element.find('span', class_='a-icon-alt').text.split(' ')[0])

                reviews.append({'Text': review_text, 'Score': review_score})

        if len(reviews) >= total_reviews:
            break

    return reviews[:total_reviews]

def save_reviews_to_excel(reviews, excel_file):
    df = pd.DataFrame(reviews)
    df.to_excel(excel_file, index=False)

if __name__ == '__main__':
    product_url = 'https://www.amazon.com/product-reviews/B07ZLRGJ94/ref=atv_dp_cr_see_all?ie=UTF8&reviewerType=all_reviews'
    total_reviews = 10
    excel_file = 'outputs.xlsx'

    scraped_reviews = scrape_amazon_reviews(product_url, total_reviews)
    save_reviews_to_excel(scraped_reviews, excel_file)

    print(f'{len(scraped_reviews)} reviews saved to {excel_file}')
