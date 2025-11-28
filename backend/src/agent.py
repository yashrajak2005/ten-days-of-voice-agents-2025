import logging
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("grocery_agent")
load_dotenv(".env.local")

class GroceryAgent(Agent):
    def __init__(self, catalog_path: str):
        super().__init__(
            instructions=(
                "You are a friendly and helpful Grocery Ordering Assistant for 'FreshPick Market'. "
                "Your goal is to help users browse the catalog, add items to their cart, and place orders. "
                "1. Greet the user warmly and ask how you can help them today. "
                "2. You can search the catalog for items. If a user asks for something generic like 'bread', list the available options. "
                "3. When a user wants to add an item, ask for the quantity if not specified. "
                "4. You can intelligently add ingredients for common dishes (e.g., 'ingredients for a sandwich'). "
                "5. Always confirm actions verbally (e.g., 'I've added 2 apples to your cart'). "
                "6. If a user asks 'What's in my cart?', list the items and the current total. "
                "7. When the user says they are done or wants to place the order, confirm the final total and use the 'place_order' tool. "
                "8. You can also track orders. If a user asks 'Where is my order?' or 'List my past orders', use the appropriate tools. "
                "9. Be polite, concise, and helpful. "
            )
        )
        self.catalog_path = catalog_path
        self.catalog: List[Dict] = []
        self.cart: List[Dict] = []  # List of {item: dict, quantity: int, notes: str}
        self._load_catalog()

        # Simple recipe mapping for "ingredients for X"
        self.recipes = {
            "sandwich": ["Whole Wheat Bread", "Peanut Butter", "Strawberry Jam"],
            "peanut butter sandwich": ["Whole Wheat Bread", "Peanut Butter"],
            "pasta": ["Spaghetti Pasta", "Marinara Sauce"],
            "spaghetti": ["Spaghetti Pasta", "Marinara Sauce"],
        }

    def _load_catalog(self):
        try:
            if os.path.exists(self.catalog_path):
                with open(self.catalog_path, "r", encoding="utf-8") as f:
                    self.catalog = json.load(f)
                logger.info(f"Loaded {len(self.catalog)} items from {self.catalog_path}")
            else:
                logger.warning(f"Catalog file not found at {self.catalog_path}")
                self.catalog = []
        except Exception as e:
            logger.error(f"Error loading catalog: {e}")
            self.catalog = []

    def _find_item_by_name(self, name: str) -> Optional[Dict]:
        name_lower = name.lower()
        # Exact match first
        for item in self.catalog:
            if item["name"].lower() == name_lower:
                return item
        # Partial match
        for item in self.catalog:
            if name_lower in item["name"].lower():
                return item
        return None

    @function_tool
    async def get_catalog_items(self, context: RunContext, category: Optional[str] = None):
        """List items in the catalog, optionally filtered by category."""
        if category:
            items = [item["name"] for item in self.catalog if item["category"].lower() == category.lower()]
            if not items:
                return f"No items found in category '{category}'."
            return f"In {category}, we have: {', '.join(items)}."
        else:
            # List a few categories and examples
            categories = list(set(item["category"] for item in self.catalog))
            return f"We have items in the following categories: {', '.join(categories)}. Ask me about specific items!"

    @function_tool
    async def add_to_cart(self, context: RunContext, item_name: str, quantity: int = 1, notes: str = ""):
        """Add an item to the cart."""
        item = self._find_item_by_name(item_name)
        if not item:
            return f"I couldn't find '{item_name}' in our catalog."
        
        # Check if item already in cart
        for cart_item in self.cart:
            if cart_item["item"]["id"] == item["id"]:
                cart_item["quantity"] += quantity
                if notes:
                    cart_item["notes"] = f"{cart_item.get('notes', '')} {notes}".strip()
                return f"Updated {item['name']} quantity to {cart_item['quantity']}."

        self.cart.append({"item": item, "quantity": quantity, "notes": notes})
        return f"Added {quantity} {item['name']} to your cart."

    @function_tool
    async def remove_from_cart(self, context: RunContext, item_name: str):
        """Remove an item from the cart."""
        item = self._find_item_by_name(item_name)
        if not item:
            return f"I couldn't find '{item_name}' to remove."
        
        for i, cart_item in enumerate(self.cart):
            if cart_item["item"]["id"] == item["id"]:
                removed = self.cart.pop(i)
                return f"Removed {removed['item']['name']} from your cart."
        
        return f"You don't have '{item_name}' in your cart."

    @function_tool
    async def get_cart_contents(self, context: RunContext):
        """List the contents of the cart and the total price."""
        if not self.cart:
            return "Your cart is empty."
        
        lines = []
        total = 0.0
        for cart_item in self.cart:
            item = cart_item["item"]
            qty = cart_item["quantity"]
            cost = item["price"] * qty
            total += cost
            note = f" ({cart_item['notes']})" if cart_item['notes'] else ""
            lines.append(f"{qty}x {item['name']}{note} - ${cost:.2f}")
        
        return f"Your cart contains:\n" + "\n".join(lines) + f"\nTotal: ${total:.2f}"

    @function_tool
    async def add_ingredients_for_dish(self, context: RunContext, dish_name: str, quantity: int = 1):
        """Add all ingredients for a specific dish to the cart."""
        dish_lower = dish_name.lower()
        ingredients = None
        
        # Check exact match
        if dish_lower in self.recipes:
            ingredients = self.recipes[dish_lower]
        else:
            # Check partial match
            for key in self.recipes:
                if key in dish_lower or dish_lower in key:
                    ingredients = self.recipes[key]
                    break
        
        if not ingredients:
            return f"I don't have a recipe for '{dish_name}' yet. Please add items individually."
        
        added_items = []
        for ing_name in ingredients:
            await self.add_to_cart(context, ing_name, quantity)
            added_items.append(ing_name)
            
        return f"Added ingredients for {dish_name}: {', '.join(added_items)}."

    @function_tool
    async def place_order(self, context: RunContext):
        """Place the order and save it to a file."""
        if not self.cart:
            return "Your cart is empty. I cannot place an order."
        
        total = sum(item["item"]["price"] * item["quantity"] for item in self.cart)
        
        order_data = {
            "timestamp": datetime.now().isoformat(),
            "items": [
                {
                    "id": i["item"]["id"],
                    "name": i["item"]["name"],
                    "price": i["item"]["price"],
                    "quantity": i["quantity"],
                    "notes": i["notes"]
                }
                for i in self.cart
            ],
            "total": total,
            "status": "placed"
        }
        
        # Save to single order.json file
        orders_file = os.path.join(os.path.dirname(self.catalog_path), "order.json")
        
        try:
            orders = []
            if os.path.exists(orders_file):
                with open(orders_file, "r", encoding="utf-8") as f:
                    orders = json.load(f)
            
            # Generate a simple ID if not present (using timestamp for uniqueness)
            order_id = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            order_data["id"] = order_id
            
            orders.append(order_data)
            
            with open(orders_file, "w", encoding="utf-8") as f:
                json.dump(orders, f, indent=2)
            
            # Clear cart
            self.cart = []
            return f"Order placed successfully! Total was ${total:.2f}. Your order ID is {order_id}."
        except Exception as e:
            logger.error(f"Failed to save order: {e}")
            return "I'm sorry, there was an error saving your order. Please try again."

    @function_tool
    async def get_order_status(self, context: RunContext, order_id: str):
        """Check the status of a specific order by its ID."""
        orders_file = os.path.join(os.path.dirname(self.catalog_path), "order.json")
        
        if not os.path.exists(orders_file):
            return "No orders have been placed yet."
        
        try:
            with open(orders_file, "r", encoding="utf-8") as f:
                orders = json.load(f)
            
            # Find order
            order = next((o for o in orders if o.get("id") == order_id or order_id in o.get("id", "")), None)
            
            if not order:
                return f"I couldn't find an order with ID {order_id}."
            
            # Mock status logic based on time elapsed
            order_time = datetime.fromisoformat(order["timestamp"])
            elapsed = (datetime.now() - order_time).total_seconds() / 60 # minutes
            
            status = "Received"
            if elapsed > 10:
                status = "Delivered"
            elif elapsed > 5:
                status = "Out for Delivery"
            elif elapsed > 1:
                status = "Preparing"
                
            return f"Order {order_id} status: {status} (Placed on {order_time.strftime('%Y-%m-%d %H:%M')}). Total: ${order['total']:.2f}."
            
        except Exception as e:
            logger.error(f"Error reading orders: {e}")
            return "I couldn't read the order history."

    @function_tool
    async def list_past_orders(self, context: RunContext):
        """List the 5 most recent orders."""
        orders_file = os.path.join(os.path.dirname(self.catalog_path), "order.json")
        if not os.path.exists(orders_file):
            return "No orders found."
            
        try:
            with open(orders_file, "r", encoding="utf-8") as f:
                orders = json.load(f)
            
            # Sort by timestamp descending
            orders.sort(key=lambda x: x["timestamp"], reverse=True)
            
            recent = orders[:5]
            if not recent:
                return "No recent orders found."
                
            response = "Here are your recent orders:\n"
            for order in recent:
                timestamp = datetime.fromisoformat(order["timestamp"]).strftime('%Y-%m-%d %H:%M')
                response += f"- {order.get('id', 'Unknown ID')}: ${order['total']:.2f} ({timestamp})\n"
                    
            return response
        except Exception as e:
            logger.error(f"Error listing orders: {e}")
            return "I couldn't list your past orders."

# ----- Hooks used by the job process -----
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Initialize Grocery Catalog
    catalog_path = os.path.join(os.path.dirname(__file__), "grocery_catalog.json")
    if not os.path.exists(catalog_path):
         catalog_path = os.path.join(os.getcwd(), "src", "grocery_catalog.json")

    # Build session
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Instantiate agent
    agent = GroceryAgent(catalog_path=catalog_path)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))