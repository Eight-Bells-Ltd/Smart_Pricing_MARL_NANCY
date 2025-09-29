from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import ValidationError, BaseModel
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


@smart_pricing_api.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")


# Wrap evaluate inside a Ray task so failures don’t poison the main worker
@ray.remote
def safe_evaluate(*args, **kwargs):
    return evaluate(*args, **kwargs)


@smart_pricing_api.post("/price_calculation")
async def calculate_price(payload: ServicesPayload):
    try:
        services = payload.services
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
        service_name = services[0].service_id

        num_bidders = len(possible_agents)
        max_rounds = 10

        providers_min_prices, providers_max_prices, initial_prices = load_balancing(
            providers_min_prices, providers_max_prices, providers_availability
        )

        auction_result = ray.get(
            safe_evaluate.remote(
                ReverseAuctionEnv,
                model_path="models/test",
                render_mode="deploy",
                use_init_values=True,
                num_bidders=num_bidders,
                possible_agents=possible_agents,
                initial_prices=initial_prices,
                min_limit_bid=providers_min_prices,
                max_limit_bid=providers_max_prices,
                max_rounds=max_rounds,
            )
        )

        response = {
            "provider_id": auction_result["winner"],
            "price": auction_result["price"],
            "service_id": service_name,
        }
        logger.info(f"Auction successful: {response}")
        return {"services": response}

    except ValidationError as ve:
        logger.warning(f"Validation failed: {ve}")
        raise HTTPException(status_code=422, detail=f"Validation failed: {ve}")
    except Exception as e:
        logger.error("Unexpected error in /price_calculation", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")



# ---- Exception Handlers ----
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


@smart_pricing_api.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception at {request.url}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "Unexpected error", "details": str(exc)},
    )



# ---- Run with multiple workers for resilience ----
# Example: uvicorn main:smart_pricing_api --host 0.0.0.0 --port 8000 --workers 4
