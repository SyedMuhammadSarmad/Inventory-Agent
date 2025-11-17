import os
from datetime import datetime
from typing import List
from agents import Agent, ItemHelpers, function_tool,Runner,set_default_openai_client,set_tracing_export_api_key,set_default_openai_api
from openai import AsyncOpenAI
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from openai.types.responses import ResponseInputItemParam
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

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
Trace_key = os.getenv("TRACING_API_KEY")

client = AsyncOpenAI(base_url = BASE_URL, api_key = API_KEY )
set_default_openai_client(client,use_for_tracing=False)
set_tracing_export_api_key(Trace_key)
# set_default_openai_api("chat_completions",)


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


class CreateReorderInput(BaseModel):
    item_id: int
    quantity: int
    requested_by: str = "Inventory_agent"

class ApproveReorderInput(BaseModel):
    reorder_id: int
    approve: bool
    approver: str = "finance_agent"

class CreatePaymentInput(BaseModel):
    reorder_id: int

class message(BaseModel):
    msg:str


@function_tool
def get_low_stock_items_tool():
    """Return all items where quantity is below reorder level."""
    from schema import engine
    from sqlmodel import Session, select

    with Session(engine) as session:
        items = session.exec(select(InventoryItem)).all()

        low_stock = [
            {
                "id": i.id,
                "name": i.name,
                "quantity": i.quantity,
                "reorder_level": i.reorder_level
            }
            for i in items if i.quantity < i.reorder_level
        ]

    return low_stock

@function_tool
def create_reorder_tool(data: CreateReorderInput) -> dict:
    """Create a reorder entry for an item."""
    from schema import engine
    from sqlmodel import Session, select

    with Session(engine) as session:

        item = session.get(InventoryItem, data.item_id)
        if not item:
            return {"error": "Item not found"}

        total_cost = data.quantity * item.price_per_unit

        reorder = Reorder(
            item_id=data.item_id,
            quantity_requested=data.quantity,
            unit_price=item.price_per_unit,
            total_cost=total_cost,
            requested_by=data.requested_by,
        )

        session.add(reorder)
        session.commit()
        session.refresh(reorder)

        return {"message": "Reorder created", "reorder": reorder}


@function_tool
def approve_reorder_tool(data: ApproveReorderInput):
    """Approve or reject a reorder."""
    from schema import engine
    from sqlmodel import Session

    with Session(engine) as session:

        reorder = session.get(Reorder, data.reorder_id)
        if not reorder:
            return {"error": "Reorder not found"}

        if data.approve:
            reorder.status = "approved"
            reorder.approved_by = data.approver
            reorder.approved_at = datetime.utcnow()
        else:
            reorder.status = "rejected"

        session.commit()

        return {"message": f"Reorder {reorder.status}"}


@function_tool
def create_payment_tool(data: CreatePaymentInput):
    """Create payment for an approved reorder."""
    from schema import engine
    from sqlmodel import Session, select

    with Session(engine) as session:

        reorder = session.get(Reorder, data.reorder_id)
        if not reorder or reorder.status != "approved":
            return {"error": "Reorder must be approved first"}

        payment = Payment(
            reorder_id=data.reorder_id,
            amount=reorder.total_cost,
            status="paid",
            paid_at=datetime.utcnow(),
        )
        session.add(payment)

        # Update budget
        budget = session.exec(select(Budget).limit(1)).first()
        if budget:
            budget.spent_amount += reorder.total_cost
            session.add(budget)

        reorder.status = "paid"
        session.add(reorder)

        session.commit()

        return {"message": "Payment successful", "payment": payment}

class FinanceOutput(BaseModel):
    approved: List[int]
    paid: List[int]
    total_spent: float
    summary: str
finance_agent = Agent(
    name="finance agent",
    instructions=
f"""{RECOMMENDED_PROMPT_PREFIX}
You are the Finance Agent.
you can approve reorders and create payments
"""
,
    tools=[approve_reorder_tool,
           create_payment_tool,
        ], 
    
    handoff_description="Finance agent which handle finance queries like approve reorder or create payments",

    output_type=FinanceOutput,
    model='gpt-4o')

class InventoryOutput(BaseModel):
    low_stock_count: int
    reorders_created: List[int]
    total_cost: float
    summary: str

inventory_agent = Agent(
    name="Inventory Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are the Inventory Agent.You can check low stock items and create re-order requests
""",
    tools=[get_low_stock_items_tool,
        create_reorder_tool,
        ], 

    handoff_description="inventory agent which can check low stock items and create proper reorder requests",

    model='gpt-4o',

    output_type=InventoryOutput,
 
)


manager_agent = Agent(name="Manager agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are a manager agent you always use the proper tools for according query
""",
    tools=[
        finance_agent.as_tool(
            tool_name="finance_tool",
            tool_description="do the finance work"
        ),
        inventory_agent.as_tool(
            tool_name="inventory_tool",
            tool_description="do inventory management work"
        )],
    model='gpt-4o')

@app.post("/agent/chat")
async def main(req: message, session: Session = Depends(get_session)):

    result = Runner.run_streamed(manager_agent, req.msg,)
    print("=== Run starting ===")

    async for event in result.stream_events():
        # We'll ignore the raw responses event deltas
        if event.type == "raw_response_event":
            continue
        # When the agent updates, print that
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
            continue
        # When items are generated, print them
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- Tool was called")
                tool_input = event.item.to_input_item()
                print(f"Tool name: {tool_input}")
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
            else:
                pass  # Ignore other event types

    print("=== Run complete ===")
    return result.final_output



    
