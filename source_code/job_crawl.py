import logging
import pandas as pd
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData, EventMetrics
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import (
    RelevanceFilters,
    TimeFilters,
    TypeFilters,
)

logging.basicConfig(level=logging.INFO)

JOB_TITLE = "Data Scientist"
LOCATION = ["Viet Nam","Ha Noi","Ho Chi Minh"]
job_postings = []



def on_data(data: EventData):

    print(
        "[ON_DATA]",
        data.title,
        data.company,
        data.company_link,
        data.date,
        data.link,
        data.insights,
        len(data.description),
    )
    job_postings.append(
        [
            data.job_id,
            data.location,
            data.title,
            data.company,
            data.date,
            data.link,
            data.description,
        ]
    )

    df = pd.DataFrame(
        job_postings,
        columns=[
            "Job_ID",
            "Location",
            "Title",
            "Company",
            "Date",
            "Link",
            "Description",
        ],
    )
    df.to_csv("data/jobs.csv")
def on_metrics(metrics: EventMetrics):
    print("[ON_METRICS]", str(metrics))
    
def on_error(error):
    print("[ON_ERROR]", error)
    
def on_end():
    print("[ON_END]")


def initialise_scraper():
    scraper = LinkedinScraper(
        chrome_executable_path=None,  
        chrome_binary_location=None, 
        chrome_options=None,  
        headless=True,  
        max_workers=1,  
        slow_mo=0.5,  
        page_load_timeout=40,  
    )
    scraper.on(Events.DATA, on_data)
    scraper.on(Events.ERROR, on_error)
    scraper.on(Events.END, on_end)

    return scraper


def initialise_query(job_title: str, locations):
    queries = [
        Query(options=QueryOptions(limit=100)), 
        Query(
            query=job_title,
            options=QueryOptions(
                locations=locations,
                apply_link=True,  
                skip_promoted_jobs=True,  
                page_offset=10,  
                limit=10,
                filters=QueryFilters(
                    
                    relevance=RelevanceFilters.RECENT,
                    time=TimeFilters.MONTH,
                    type=[TypeFilters.FULL_TIME],
                    
                ),
            ),
        ),
    ]
    return queries

def scrape_jobs(job_title: str, locations: list):
    scraper = initialise_scraper()
    queries = initialise_query(job_title, locations)
    scraper.run(queries)
    return scraper

if __name__ == "__main__":
    scraper = scrape_jobs(JOB_TITLE, LOCATION)