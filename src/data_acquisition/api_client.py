import requests
import time
import threading
import numpy as np
from typing import Optional, Dict, Any

from ..utils import (
    API_RATE_LIMIT, API_RETRY_ATTEMPTS, API_RETRY_DELAY,
    print_file_operation, format_number
)


class ApiManager:
    def __init__(self, api_key: str, requests_per_minute: int = API_RATE_LIMIT):
        self.api_key = api_key
        self.rate_limit = requests_per_minute
        self.cache = {}
        self.call_times = []
        self.lock = threading.Lock()
        self.requests_count = 0
        self.esg_request_counter = 0
        self.profile_request_counter = 0

    def _check_rate_limit(self):
        now = time.time()
        with self.lock:
            self.call_times = [t for t in self.call_times if now - t < 60]
            self.requests_count += 1
            if len(self.call_times) < self.rate_limit:
                self.call_times.append(now)
                return 0
            else:
                oldest_call = min(self.call_times)
                wait_time = 60 - (now - oldest_call) + 0.5
                return wait_time

    def get(self, url: str) -> Optional[Dict[str, Any]]:
        if url in self.cache:
            return self.cache[url]

        wait_time = self._check_rate_limit()
        if wait_time > 0:
            time.sleep(wait_time)

        max_retries = API_RETRY_ATTEMPTS
        for retry in range(max_retries):
            try:
                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()
                    self.cache[url] = data
                    return data
                elif response.status_code == 429:
                    retry_delay = (retry + 1) * API_RETRY_DELAY
                    print(f"Rate limit hit (429). Waiting {retry_delay}s before retry {retry+1}/{max_retries}")
                    time.sleep(retry_delay)
                else:
                    if retry < max_retries - 1:
                        print(f"API Error: {response.status_code} for {url}, retrying...")
                        time.sleep(1)
                    else:
                        print(f"API Error: {response.status_code} for {url}, giving up after {max_retries} attempts")
                        return None
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Request Exception: {e}, retrying...")
                    time.sleep(1)
                else:
                    print(f"Request Exception: {e}, giving up after {max_retries} attempts")
                    return None

        return None

    def get_esg_data(self, ticker: str) -> list:
        url = f"https://financialmodelingprep.com/api/v4/esg-environmental-social-governance-data?symbol={ticker}&apikey={self.api_key}"

        if url in self.cache:
            return self.cache[url]

        with self.lock:
            self.esg_request_counter += 1
            if self.esg_request_counter % 5 == 0:
                time.sleep(1)

        wait_time = self._check_rate_limit()
        if wait_time > 0:
            time.sleep(wait_time)

        try:
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    esg_time_series = []
                    for item in data:
                        esg_entry = {
                            'date': item.get('date', None),
                            'environmentalScore': item.get('environmentalScore', np.nan),
                            'socialScore': item.get('socialScore', np.nan),
                            'governanceScore': item.get('governanceScore', np.nan),
                            'ESGScore': item.get('ESGScore', np.nan)
                        }
                        esg_time_series.append(esg_entry)

                    self.cache[url] = esg_time_series
                    return esg_time_series
                return []
            else:
                print(f"Error fetching ESG data for {ticker}: {response.status_code}")
                return []
        except Exception as e:
            print(f"Exception while fetching ESG data for {ticker}: {e}")
            return []

    def get_esg_risk_rating(self, ticker: str) -> Dict[str, Any]:
        url = f"https://financialmodelingprep.com/api/v4/esg-environmental-social-governance-data-ratings?symbol={ticker}&apikey={self.api_key}"

        if url in self.cache:
            return self.cache[url]

        with self.lock:
            self.esg_request_counter += 1
            if self.esg_request_counter % 5 == 0:
                time.sleep(1)

        wait_time = self._check_rate_limit()
        if wait_time > 0:
            time.sleep(wait_time)

        try:
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    filtered_data = {
                        'ESGRiskRating': data[0].get('ESGRiskRating', np.nan)
                    }
                    self.cache[url] = filtered_data
                    return filtered_data
                return {}
            else:
                print(f"Error fetching ESG risk rating for {ticker}: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Exception while fetching ESG risk rating for {ticker}: {e}")
            return {}

    def get_company_profile(self, ticker: str) -> Dict[str, Any]:
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={self.api_key}"

        if url in self.cache:
            return self.cache[url]

        with self.lock:
            self.profile_request_counter += 1
            if self.profile_request_counter % 5 == 0:
                time.sleep(1)

        wait_time = self._check_rate_limit()
        if wait_time > 0:
            time.sleep(wait_time)

        try:
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    filtered_data = {
                        'sector': data[0].get('sector', None),
                        'ipoDate': data[0].get('ipoDate', None)
                    }
                    self.cache[url] = filtered_data
                    return filtered_data
                return {}
            else:
                print(f"Error fetching company profile for {ticker}: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Exception while fetching company profile for {ticker}: {e}")
            return {}

    def get_cache_stats(self) -> Dict[str, int]:
        return {
            'total_cached_urls': len(self.cache),
            'total_requests_made': self.requests_count,
            'esg_requests': self.esg_request_counter,
            'profile_requests': self.profile_request_counter
        }

    def clear_cache(self) -> None:
        self.cache.clear()
        print("API cache cleared")

    def print_stats(self) -> None:
        stats = self.get_cache_stats()
        print(f"API Manager Statistics:")
        print(f"  Total requests: {format_number(stats['total_requests_made'])}")
        print(f"  ESG requests: {format_number(stats['esg_requests'])}")
        print(f"  Profile requests: {format_number(stats['profile_requests'])}")
        print(f"  Cached URLs: {format_number(stats['total_cached_urls'])}")
