import os
import json
from datetime import datetime
from typing import List, Dict

from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, select
from dotenv import load_dotenv
import openai

from schema import (
    InventoryItem,
    Reorder,
    Payment,
    Budget,
    create_db_and_tables,
    get_session,
    seed_initial_data,
)


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    seed_initial_data()
    yield
    print("Application shutting down...")

app = FastAPI(title="Inventory + Finance Agents", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_low_stock_items(session: Session):
    items = session.exec(select(InventoryItem)).all()
    return [
        {"id": i.id, "name": i.name, "quantity": i.quantity, "reorder_level": i.reorder_level}
        for i in items if i.quantity < i.reorder_level
    ]

def create_reorder(session: Session, item_id: int, quantity: int, requested_by="agent"):
    item = session.get(InventoryItem, item_id)
    if not item:
        return {"error": "Item not found"}

    total_cost = quantity * item.price_per_unit
    reorder = Reorder(item_id=item_id, quantity_requested=quantity, unit_price=item.price_per_unit, total_cost=total_cost, requested_by=requested_by)
    session.add(reorder)
    session.commit()
    session.refresh(reorder)
    return {"message": "Reorder created", "reorder": reorder}

def approve_reorder(session: Session, reorder_id: int, approve: bool, approver="finance_agent"):
    reorder = session.get(Reorder, reorder_id)
    if not reorder:
        return {"error": "Reorder not found"}

    if approve:
        reorder.status = "approved"
        reorder.approved_by = approver
        reorder.approved_at = datetime.utcnow()
    else:
        reorder.status = "rejected"

    session.add(reorder)
    session.commit()
    return {"message": f"Reorder {reorder.status}"}

def create_payment(session: Session, reorder_id: int):
    reorder = session.get(Reorder, reorder_id)
    if not reorder or reorder.status != "approved":
        return {"error": "Reorder must be approved first"}

    payment = Payment(reorder_id=reorder_id, amount=reorder.total_cost, status="paid", paid_at=datetime.utcnow())
    session.add(payment)

    budget = session.exec(select(Budget).limit(1)).first()
    if budget:
        budget.spent_amount += reorder.total_cost
        session.add(budget)

    reorder.status = "paid"
    session.add(reorder)
    session.commit()
    return {"message": "Payment successful", "payment": payment}


class AgentChatRequest(BaseModel):
    agent: str
    messages: List[Dict[str, str]]

@app.post("/agent/chat")
def chat_with_agent(req: AgentChatRequest, session: Session = Depends(get_session)):
    system_prompt = (
        "You are an AI agent that helps manage inventory and finance tasks. "
        "Decide when to call appropriate backend functions to perform actions."
    )

    messages = [{"role": "system", "content": system_prompt}] + req.messages

    functions = [
        {
            "name": "get_low_stock_items",
            "description": "Retrieve all items below their reorder level",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "create_reorder",
            "description": "Create a reorder for an item",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "integer"},
                    "quantity": {"type": "integer"},
                },
                "required": ["item_id", "quantity"],
            },
        },
        {
            "name": "approve_reorder",
            "description": "Approve or reject a reorder request",
            "parameters": {
                "type": "object",
                "properties": {
                    "reorder_id": {"type": "integer"},
                    "approve": {"type": "boolean"},
                },
                "required": ["reorder_id", "approve"],
            },
        },
        {
            "name": "create_payment",
            "description": "Process payment for an approved reorder",
            "parameters": {
                "type": "object",
                "properties": {
                    "reorder_id": {"type": "integer"},
                },
                "required": ["reorder_id"],
            },
        },
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=functions,
        function_call="auto",
    )

    message = response["choices"][0]["message"]

    if "function_call" in message:
        fname = message["function_call"]["name"]
        args = json.loads(message["function_call"].get("arguments", "{}"))

        if fname == "get_low_stock_items":
            return get_low_stock_items(session)
        elif fname == "create_reorder":
            return create_reorder(session, **args)
        elif fname == "approve_reorder":
            return approve_reorder(session, **args)
        elif fname == "create_payment":
            return create_payment(session, **args)
        else:
            return {"error": f"Unknown function: {fname}"}

    return {"response": message["content"]}
