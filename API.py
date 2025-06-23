from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import ValidationError
import json
from pydantic import BaseModel
from typing import List
from ray import init, available_resources
from uvicorn import run

from evaluation.evaluate import evaluate
from environment.reverse_auction_env import ReverseAuctionEnv
from utils.helpers import load_balancing


smart_pricing_api = FastAPI(
    title="Smart Pricing",
    description="Global Description",
    version="0.0.1"
)

init(
    include_dashboard=False,
    ignore_reinit_error=True,
    log_to_driver=False,  # Disables logging to the driver
    _system_config={
        "metrics_report_interval_ms": 0,  # Disables periodic metrics reporting
    }
)
# print(available_resources())

class Service(BaseModel):
    provider_id: str
    minprice: float
    maxprice: float
    availability: float
    service_id: str


class ServicesPayload(BaseModel):
    services: List[Service]

@smart_pricing_api.get("/")
async def docs_redirect():
    return RedirectResponse(url='/docs')

@smart_pricing_api.post("/price_calculation")
async def calculate_price(payload: ServicesPayload):
    services = payload.services
    possible_agents = []
    providers_min_prices = []
    providers_max_prices = []
    providers_availability = []
    service_name = services[0].service_id #service_id is the same for all providers
    for service in services:
        possible_agents.append(service.provider_id)
        providers_max_prices.append(service.maxprice)
        providers_min_prices.append(service.minprice)
        providers_availability.append(service.availability)
    num_bidders = len(possible_agents)
    max_rounds = 10

    providers_min_prices, providers_max_prices, initial_prices = load_balancing(providers_min_prices, providers_max_prices, providers_availability)
    # initial_prices = [(max + min) / 2 for max, min in zip(providers_max_prices,providers_min_prices)]

    auction_result = evaluate(ReverseAuctionEnv, model_path="models/test", render_mode="deploy", use_init_values = True,
             num_bidders=num_bidders, possible_agents=possible_agents, initial_prices=initial_prices,
             min_limit_bid=providers_min_prices, max_limit_bid=providers_max_prices, max_rounds=max_rounds)


    response= {"provider_id":auction_result['winner'], "price":auction_result['price'], "service_id":service_name}
    print(response)
    return {"services": response}

@smart_pricing_api.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"message": "Invalid payload format", "details": exc.errors()},
    )

@smart_pricing_api.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )


# if __name__ == "__main__":
#     run(smart_pricing_api, host="127.0.0.1", port=8000)
