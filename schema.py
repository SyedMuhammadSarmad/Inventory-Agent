import os
from datetime import datetime
from typing import Optional, Generator
from sqlmodel import SQLModel, Field, Session, create_engine, select


class InventoryItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    sku: str
    quantity: int = 0
    reorder_level: int = 5
    price_per_unit: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class Reorder(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    item_id: int
    quantity_requested: int
    unit_price: float
    total_cost: float
    status: str = "pending"
    requested_by: str = "agent"
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


class Payment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    reorder_id: int
    amount: float
    method: str = "bank"
    status: str = "pending"
    paid_at: Optional[datetime] = None


class Budget(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    category: str = "general"
    allocated_amount: float = 0.0
    spent_amount: float = 0.0
    month: str = Field(default_factory=lambda: datetime.utcnow().strftime("%b-%Y"))


DB_URL = "sqlite:///./hackathon_inventory.db"
engine = create_engine(DB_URL, echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator:
    with Session(engine) as session:
        yield session


def seed_initial_data():
    with Session(engine) as session:
        if session.exec(select(InventoryItem)).first():
            return

        item1 = InventoryItem(name="Blue T-Shirt", sku="TSHIRT-001", quantity=8, reorder_level=10, price_per_unit=12.5)
        item2 = InventoryItem(name="Black Jeans", sku="JEANS-031", quantity=20, reorder_level=5, price_per_unit=25.0)
        item3 = InventoryItem(name="Black shirt", sku="shirt-012", quantity=20, reorder_level=10, price_per_unit=27.0)
        item4 = InventoryItem(name="Black T-shirt", sku="JEANS-002", quantity=20, reorder_level=25, price_per_unit=35.0)
        budget = Budget(category="general", allocated_amount=10000.0, spent_amount=250.0)

        session.add_all([item1, item2, item3, item4, budget])
        session.commit()
