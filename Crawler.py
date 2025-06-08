import scrapy
import json
import pandas as pd
from scrapy.crawler import CrawlerProcess
import time

class WatchbaseSpider(scrapy.Spider):
    name = 'watchbase'
    allowed_domains = ['watchbase.com']
    base_url = 'https://www.watchbase.com'
    
    brands = ['rolex', 'omega', 'tag-heuer', 'tudor', 'longines', 'iwc', 'breitling', 'cartier', 'panerai', 'patek-philippe']
    max_models = 50  
    
    def __init__(self, *args, **kwargs):
        super(WatchbaseSpider, self).__init__(*args, **kwargs)
        if 'brands' in kwargs:
            self.brands = kwargs.get('brands').split(',')
        if 'max_models' in kwargs:
            self.max_models = int(kwargs.get('max_models'))
    
    def start_requests(self):
        """Generate the initial requests for each brand"""
        for brand in self.brands:
            yield scrapy.Request(
                url=f'{self.base_url}/{brand}',
                callback=self.parse_brand_page,
                meta={'brand': brand}
            )
    
    def parse_brand_page(self, response):
        """Parse the brand page to extract collection links"""
        brand = response.meta.get('brand')
        self.logger.info(f"Processing brand: {brand}")
        
        collection_links = response.css('h2.title > a::attr(href)').getall()
        self.logger.info(f"Found {len(collection_links)} collections for {brand}")
        
        for col_link in collection_links:
            yield scrapy.Request(
                url=col_link, 
                callback=self.parse_collection_page,
                meta={'brand': brand}
            )
    
    def parse_collection_page(self, response):
        """Parse the collection page to extract model links"""
        brand = response.meta.get('brand')
        
        model_links = response.css('a.item-block.watch-block::attr(href)').getall()
        model_links = [link for link in model_links if brand in link]
        self.logger.info(f"Found {len(model_links)} models in collection: {response.url}")
        
        for link in model_links[:self.max_models]:
            yield scrapy.Request(
                url=link,
                callback=self.parse_model_page,
                meta={'brand': brand}
            )
    
    def parse_model_page(self, response):
        """Parse the watch model details page"""
        self.logger.info(f"Parsing model: {response.url}")
        
        def get_text_after_th(label):
            element = response.xpath(f"//th[contains(text(),'{label}')]/following-sibling::td[1]")
            if element:
                anchor = element.css('a::text').get()
                if anchor:
                    return anchor.strip()
                return element.css('::text').get('').strip()
            return ""
        
        item = {
            'Url': response.url,
            'Brand': get_text_after_th('Brand:'),
            'Family': get_text_after_th('Family:'),
            'Reference': get_text_after_th('Reference:'),
            'Name': get_text_after_th('Name:'),
            'Movement': get_text_after_th('Movement:'),
            'Produced': get_text_after_th('Produced:'),
            'Limited': get_text_after_th('Limited:'),
            'Case_Material': get_text_after_th('Material:'),
            'Glass': get_text_after_th('Glass:'),
            'Case_Back': get_text_after_th('Back:'),
            'Case_Shape': get_text_after_th('Shape:'),
            'Case_Diameter': get_text_after_th('Diameter:'),
            'Lug_Width': get_text_after_th('Lug Width:'),
            'Water_Resistance': get_text_after_th('W/R:'),
            'Dial_Color': get_text_after_th('Color:'),
            'Dial_Finish': get_text_after_th('Finish:'),
            'Dial_Indexes': get_text_after_th('Indexes:'),
            'Dial_Hands': get_text_after_th('Hands:')
        }
        
        price_url = response.css('canvas#pricechart::attr(data-url)').get()
        if price_url:
            yield scrapy.Request(
                url=price_url,
                callback=self.parse_price_data,
                meta={'item': item}
            )
        else:
            item['Price'] = "N/A"
            yield item
    
    def parse_price_data(self, response):
        """Parse the price JSON data"""
        item = response.meta.get('item')
        
        try:
            data = json.loads(response.text)
            prices = data.get('datasets', [{}])[0].get('data', [])
            
            price = "N/A"
            for p in reversed(prices):
                if p is not None:
                    price = p
                    break
                    
            item['Price'] = price
        except Exception as e:
            self.logger.error(f"Error parsing price data: {e}")
            item['Price'] = "N/A"
        
        yield item

class WatchbaseDataPipeline:
    """Pipeline to save the scraped items to a CSV file"""
    
    def __init__(self):
        self.items = []
    
    def process_item(self, item, spider):
        self.items.append(item)
        return item
    
    def close_spider(self, spider):
        if self.items:
            df = pd.DataFrame(self.items)
            df.to_csv("watchbase_data_raw_scrapy.csv", index=False)
            spider.logger.info(f"Saved {len(self.items)} watch models to watchbase_data_raw_scrapy.csv")

def run_spider():
    """Run the spider with custom settings"""
    start_time = time.time()
    
    settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 1,  
        'DOWNLOAD_DELAY': 2,  
        'ITEM_PIPELINES': {
            '__main__.WatchbaseDataPipeline': 300,
        },
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 2,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 1.0,
        'LOG_LEVEL': 'INFO',
    }
    
    process = CrawlerProcess(settings)
    process.crawl(WatchbaseSpider)
    process.start()  
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    run_spider()
