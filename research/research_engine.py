
from googleapiclient.discovery import build
from scholarly import scholarly
import arxiv
from bs4 import BeautifulSoup
import requests
import asyncio
import aiohttp
from typing import List, Dict, Any
import pandas as pd

class ResearchEngine:
    def __init__(self, config: Config):
        self.config = config
        self.google_service = build(
            "customsearch", "v1",
            developerKey=config.GOOGLE_API_KEY
        )
        
    async def gather_research(self, query: str, sources: List[str]) -> Dict[str, List[Dict]]:
        """Gather research from multiple sources concurrently"""
        tasks = []
        if "news" in sources:
            tasks.append(self.fetch_news(query))
        if "academic" in sources:
            tasks.append(self.fetch_academic_papers(query))
        if "financial" in sources:
            tasks.append(self.fetch_financial_reports(query))
            
        results = await asyncio.gather(*tasks)
        return {
            source: result for source, result in zip(sources, results)
        }
        
    async def fetch_news(self, query: str) -> List[Dict]:
        """Fetch news from multiple sources"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_google_news(session, query),
                self._fetch_financial_times(session, query),
                self._fetch_reuters(session, query)
            ]
            results = await asyncio.gather(*tasks)
            return [item for sublist in results for item in sublist]
            
    async def fetch_academic_papers(self, query: str) -> List[Dict]:
        """Fetch academic papers from arXiv and Google Scholar"""
        search = arxiv.Search(
            query=query,
            max_results=10,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in search.results():
            papers.append({
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'summary': result.summary,
                'pdf_url': result.pdf_url,
                'published': result.published
            })
            
        return papers
        
    async def fetch_financial_reports(self, query: str) -> List[Dict]:
        """Fetch financial reports and SEC filings"""
        # TODO


