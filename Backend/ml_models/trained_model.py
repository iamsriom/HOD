import logging
import pickle
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Loading model...")


def get_recommendations(sku: str) -> List[str]:
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return model[sku]
    except Exception as e:
        logger.error(e)
        return []


if __name__ == "__main__":
    sku = "HO0772"
    print(get_recommendations(sku))
