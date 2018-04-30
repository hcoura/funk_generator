# -*- coding: utf-8 -*-
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Rule, CrawlSpider
from ..items import FunkItem, FunkItemLoader


class FunkSpider(CrawlSpider):
    name = 'funk'
    start_urls = ['https://www.vagalume.com.br/browse/style/funk-carioca.html']

    artist_lx = LinkExtractor(restrict_css='#browselist ol')
    songs_lx = LinkExtractor(restrict_css='.tracks')
    rules = [
        Rule(artist_lx),
        Rule(songs_lx, callback='parse_songs'),
    ]

    def parse_songs(self, response):
        il = FunkItemLoader(FunkItem(), response=response)
        il.add_css('title', '#header h1::text')
        il.add_css('text', '*[itemprop="description"] *::text')
        yield il.load_item()
