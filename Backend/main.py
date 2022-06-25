#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18

@author: sakshamagarwal
"""
import json
from typing import List

from fastapi import FastAPI

from backend.ml_models.Recommender import get_sku
from backend.ml_models.trained_model import get_recommendations

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/skulist/{product_code}")
async def read_items(product_code: str):
    sku_data = get_recommendations(product_code)
    return json.dumps(sku_data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
