from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import List
from ray import init
import ray

from evaluation.evaluate import evaluate
from environment.reverse_auction_env import ReverseAuctionEnv
from utils.helpers import load_balancing

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more details
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("smart_pricing")


smart_pricing_api = FastAPI(
    title="Smart Pricing",
    description="Global Description",
    version="0.0.1"
)

# Initialize Ray safely
init(
    include_dashboard=False,
    ignore_reinit_error=True,
    log_to_driver=False,
    _system_config={
        "metrics_report_interval_ms": 0,
    }
)

class Service(BaseModel):
    provider_id: str
    minprice: float
    maxprice: float
    availability: float
    service_id: str


class ServicesPayload(BaseModel):
    services: List[Service]

# --- Validation helper ---
def _validate_services(services: list[Service]) -> None:
    if not services:
        raise HTTPException(status_code=422, detail="services list must not be empty")

    # All services must share the same service_id
    service_id = services[0].service_id
    for s in services:
        if s.minprice > s.maxprice:
            raise HTTPException(
                status_code=422,
                detail=f"minprice > maxprice for provider {s.provider_id}"
            )

        if not (0.0 <= s.availability <= 1.0):
            raise HTTPException(
                status_code=422,
                detail=f"availability must be in [0,1] for provider {s.provider_id}"
            )

    # No duplicate provider_ids
    seen = set()
    for s in services:
        if s.provider_id in seen:
            raise HTTPException(
                status_code=422,
                detail=f"duplicate provider_id detected: {s.provider_id}"
            )
        seen.add(s.provider_id)

@smart_pricing_api.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")


@smart_pricing_api.post("/price_calculation")
async def calculate_price(payload: ServicesPayload):
    try:
        services = payload.services

        _validate_services(services)

        if len(services) == 1:
            service = services[0]
            chosen_price = (service.minprice + service.maxprice) / 2
            response = {
                "provider_id": service.provider_id,
                "price": chosen_price,
                "service_id": service.service_id,
            }
            logger.info(f"Single-service calculation successful: {response}")
            return {"services": response}

        possible_agents = [s.provider_id for s in services]
        providers_min_prices = [s.minprice for s in services]
        providers_max_prices = [s.maxprice for s in services]
        providers_availability = [s.availability for s in services]

        num_bidders = len(possible_agents)
        max_rounds = 10

        providers_min_prices, providers_max_prices, initial_prices = load_balancing(
            providers_min_prices, providers_max_prices, providers_availability
        )

        auction_result = evaluate(ReverseAuctionEnv, model_path="models/test", render_mode="deploy", use_init_values = True,
             num_bidders=num_bidders, possible_agents=possible_agents, initial_prices=initial_prices,
             min_limit_bid=providers_min_prices, max_limit_bid=providers_max_prices, max_rounds=max_rounds)

        winner_id = auction_result["winner"]
        winner_service = next((s for s in services if s.provider_id == winner_id), None)

        if not winner_service:
            raise HTTPException(status_code=500, detail="Internal error: Winner provider not found in request")

        response = {
            "provider_id": winner_id,
            "price": auction_result["price"],
            "service_id": winner_service.service_id,
        }
        logger.info(f"Auction successful: {response}")
        return {"services": response}

    except HTTPException as ve:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")



# ---- Exception Handlers ----
@smart_pricing_api.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"http_exception_handler: {exc.detail}\n in request: {request}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )


@smart_pricing_api.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(Exception)}\n in request: {request}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "Unexpected error", "details": str(exc)},
    )



# ---- Run with multiple workers for resilience ----
# Example: uvicorn main:smart_pricing_api --host 0.0.0.0 --port 8000 --workers 4
