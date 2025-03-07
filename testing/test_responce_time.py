import requests
import time
import matplotlib.pyplot as plt

url = "http://127.0.0.1:8000/price_calculation"
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

response_times = []
iterations = 100  # Number of sequential requests

for i in range(iterations):
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data)
    elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    response_times.append(elapsed_time)
    print(f"Request {i+1}: {elapsed_time:.2f} ms | Status: {response.status_code}")

avg_time = sum(response_times) / iterations
print(f"\nAverage Response Time: {avg_time:.2f} ms")

# Plotting the frequency diagram of the response times
plt.figure(figsize=(10, 6))
plt.hist(response_times, bins=10, edgecolor='black')
plt.axvline(avg_time, color='red', linestyle='dashed', linewidth=1, label=f'Average: {avg_time:.2f} ms')
plt.xlabel('Response Time (ms)')
plt.ylabel('Frequency')
plt.title('Response Time Frequency Distribution')
plt.legend()
plt.savefig('./outputs/response_time_histogram.png')
