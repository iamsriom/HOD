#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18

@author: sakshamagarwal
"""
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from Recommender import get_sku

app = FastAPI()


class Item(BaseModel):
    code: str


class SKU_list(BaseModel):
    codes: List[Item]


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/skulist/")
async def read_items():
    data = []
    sku_data = get_sku()
    for value in sku_data:
        data.append(Item(code=value))
    return SKU_list(codes=data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
