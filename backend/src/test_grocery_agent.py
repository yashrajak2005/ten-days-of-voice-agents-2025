import asyncio
import os
import json
import sys
from unittest.mock import MagicMock

# Mock livekit plugins to avoid import errors during testing
sys.modules["livekit.plugins"] = MagicMock()
sys.modules["livekit.plugins.murf"] = MagicMock()
sys.modules["livekit.plugins.silero"] = MagicMock()
sys.modules["livekit.plugins.google"] = MagicMock()
sys.modules["livekit.plugins.deepgram"] = MagicMock()
sys.modules["livekit.plugins.noise_cancellation"] = MagicMock()
sys.modules["livekit.plugins.turn_detector"] = MagicMock()
sys.modules["livekit.plugins.turn_detector.multilingual"] = MagicMock()

from agent import GroceryAgent
from livekit.agents import RunContext

# Mock RunContext
class MockRunContext(RunContext):
    def __init__(self):
        pass

async def test_grocery_agent():
    print("--- Testing Grocery Agent ---")
    
    # Path to catalog
    catalog_path = os.path.join(os.path.dirname(__file__), "grocery_catalog.json")
    if not os.path.exists(catalog_path):
         catalog_path = os.path.join(os.getcwd(), "src", "grocery_catalog.json")
    
    agent = GroceryAgent(catalog_path=catalog_path)
    ctx = MockRunContext()
    
    print(f"\n1. Catalog Loaded: {len(agent.catalog)} items")
    
    print("\n2. Testing get_catalog_items:")
    print(await agent.get_catalog_items(ctx))
    print(await agent.get_catalog_items(ctx, category="Snacks"))
    
    print("\n3. Testing add_to_cart:")
    print(await agent.add_to_cart(ctx, "Whole Wheat Bread", 2))
    print(await agent.add_to_cart(ctx, "Nonexistent Item"))
    
    print("\n4. Testing add_ingredients_for_dish:")
    print(await agent.add_ingredients_for_dish(ctx, "peanut butter sandwich"))
    
    print("\n5. Testing get_cart_contents:")
    print(await agent.get_cart_contents(ctx))
    
    print("\n6. Testing remove_from_cart:")
    print(await agent.remove_from_cart(ctx, "Whole Wheat Bread"))
    print(await agent.get_cart_contents(ctx))
    
    print("\n7. Testing place_order:")
    print(await agent.place_order(ctx))
    
    # Verify order file creation
    orders_file = os.path.join(os.path.dirname(catalog_path), "order.json")
    if os.path.exists(orders_file):
        with open(orders_file, "r") as f:
            orders = json.load(f)
        
        if orders:
            latest_order = orders[-1]
            print(f"\nOrder added to order.json. Total orders: {len(orders)}")
            print(json.dumps(latest_order, indent=2))
            
            order_id = latest_order.get("id")
            
            print("\n8. Testing get_order_status:")
            print(await agent.get_order_status(ctx, order_id))
            
            print("\n9. Testing list_past_orders:")
            print(await agent.list_past_orders(ctx))
            
        else:
            print("\nOrder file exists but is empty.")
    else:
         print("\norder.json not found.")

if __name__ == "__main__":
    asyncio.run(test_grocery_agent())
