import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

url = "http://127.0.0.1:8001/price_calculation"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}
data = {
    "services": [
        {
            "provider_id": "OTE",
            "minprice": 50.0,
            "maxprice": 100.0,
            "service_id": "service_1"
        },
        {
            "provider_id": "Vodafone",
            "minprice": 60.0,
            "maxprice": 90.0,
            "service_id": "service_2"
        },
        {
            "provider_id": "8Bells",
            "minprice": 55.0,
            "maxprice": 95.0,
            "service_id": "service_3"
        }
    ]
}


def send_request():
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data)
    elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    return elapsed_time, response.status_code, response.text


total_time = 0
iterations = 20  # Number of parallel requests
concurrent_requests = 2  # Number of concurrent requests

with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
    futures = [executor.submit(send_request) for _ in range(iterations)]

    for i, future in enumerate(as_completed(futures), 1):
        elapsed_time, status_code, response_text = future.result()
        total_time += elapsed_time
        print(f"Request {i}: {elapsed_time:.2f} ms | Status: {status_code}")

avg_time = total_time / iterations
print(f"\nAverage Response Time: {avg_time:.2f} ms")
