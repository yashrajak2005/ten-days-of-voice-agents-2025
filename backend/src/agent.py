import logging
import json
from datetime import datetime
from pathlib import Path

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
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class CoffeeOrder:
    """Maintains the state of a coffee order"""
    def __init__(self):
        self.drink_type: str | None = None
        self.size: str | None = None
        self.milk: str | None = None
        self.extras: list[str] = []
        self.name: str | None = None
    
    def to_dict(self) -> dict:
        return {
            "drinkType": self.drink_type,
            "size": self.size,
            "milk": self.milk,
            "extras": self.extras,
            "name": self.name
        }
    
    def is_complete(self) -> bool:
        """Check if all required fields are filled"""
        return all([
            self.drink_type,
            self.size,
            self.milk,
            self.name
        ])
    
    def get_missing_fields(self) -> list[str]:
        """Get list of missing required fields"""
        missing = []
        if not self.drink_type:
            missing.append("drink type")
        if not self.size:
            missing.append("size")
        if not self.milk:
            missing.append("milk preference")
        if not self.name:
            missing.append("name")
        return missing


class BaristaAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly and enthusiastic barista at StarBeam Coffee, a cozy neighborhood coffee shop.
            The user is interacting with you via voice to place their coffee order.
            
            Your personality:
            - Warm, welcoming, and conversational
            - Patient and helpful when customers are deciding
            - Excited about coffee and making recommendations
            - Professional but approachable
            
            Your job is to take a complete coffee order by gathering:
            1. Drink type (e.g., latte, cappuccino, americano, espresso, mocha, flat white, cold brew)
            2. Size (small, medium, or large)
            3. Milk preference (whole milk, skim milk, oat milk, almond milk, soy milk, or no milk)
            4. Any extras they'd like (e.g., extra shot, vanilla syrup, caramel, whipped cream, cinnamon)
            5. Name for the order
            
            Guidelines:
            - Greet the customer warmly and ask what they'd like to order
            - Ask clarifying questions ONE at a time for any missing information
            - If they mention extras, confirm and ask if they'd like anything else added
            - Once you have all the information, summarize their order and ask for confirmation
            - After confirmation, use the save_order tool to save their order
            - Keep responses conversational and concise, as if speaking naturally
            - Don't use complex formatting, emojis, or asterisks in your speech
            - If they seem unsure, offer friendly suggestions based on popular choices""",
        )
        self.current_order = CoffeeOrder()

    @function_tool
    async def save_order(self, context: RunContext, confirmed: bool):
        """Save the completed coffee order to a JSON file.
        
        Use this tool only after the customer has confirmed their complete order.
        
        Args:
            confirmed: Whether the customer has confirmed they want to place this order (must be True to save)
        """
        if not confirmed:
            return "Order not confirmed by customer. Please confirm the order before saving."
        
        if not self.current_order.is_complete():
            missing = self.current_order.get_missing_fields()
            return f"Cannot save order - missing information: {', '.join(missing)}"
        
        # Create orders directory if it doesn't exist
        orders_dir = Path("orders")
        orders_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = orders_dir / f"order_{timestamp}_{self.current_order.name}.json"
        
        # Prepare order data with metadata
        order_data = {
            "timestamp": datetime.now().isoformat(),
            "order": self.current_order.to_dict()
        }
        
        # Save to JSON file
        with open(filename, "w") as f:
            json.dump(order_data, f, indent=2)
        
        logger.info(f"Order saved to {filename}: {order_data}")
        
        return f"Perfect! Your order has been placed, {self.current_order.name}. Your {self.current_order.size} {self.current_order.drink_type} with {self.current_order.milk} will be ready shortly. Thanks for choosing StarBeam Coffee!"

    @function_tool
    async def update_drink_type(self, context: RunContext, drink_type: str):
        """Update the drink type in the current order.
        
        Args:
            drink_type: The type of coffee drink (e.g., latte, cappuccino, americano, espresso, mocha, flat white, cold brew)
        """
        self.current_order.drink_type = drink_type.lower()
        logger.info(f"Updated drink type: {drink_type}")
        return f"Got it, one {drink_type}!"

    @function_tool
    async def update_size(self, context: RunContext, size: str):
        """Update the size in the current order.
        
        Args:
            size: The size of the drink (small, medium, or large)
        """
        self.current_order.size = size.lower()
        logger.info(f"Updated size: {size}")
        return f"Great, {size} size!"

    @function_tool
    async def update_milk(self, context: RunContext, milk: str):
        """Update the milk preference in the current order.
        
        Args:
            milk: The milk preference (whole milk, skim milk, oat milk, almond milk, soy milk, or no milk)
        """
        self.current_order.milk = milk.lower()
        logger.info(f"Updated milk: {milk}")
        return f"Perfect, {milk}!"

    @function_tool
    async def add_extra(self, context: RunContext, extra: str):
        """Add an extra item to the current order.
        
        Args:
            extra: An extra item to add (e.g., extra shot, vanilla syrup, caramel, whipped cream, cinnamon)
        """
        extra_lower = extra.lower()
        if extra_lower not in self.current_order.extras:
            self.current_order.extras.append(extra_lower)
            logger.info(f"Added extra: {extra}")
            return f"Added {extra} to your order!"
        else:
            return f"You already have {extra} in your order."

    @function_tool
    async def update_name(self, context: RunContext, name: str):
        """Update the customer's name for the order.
        
        Args:
            name: The customer's name for the order
        """
        self.current_order.name = name.strip()
        logger.info(f"Updated name: {name}")
        return f"Thanks, {name}!"

    @function_tool
    async def check_order_status(self, context: RunContext):
        """Check the current status of the order and what information is still needed.
        
        Use this to see what information has been collected and what's still missing.
        """
        order_dict = self.current_order.to_dict()
        missing = self.current_order.get_missing_fields()
        
        status = {
            "current_order": order_dict,
            "is_complete": self.current_order.is_complete(),
            "missing_fields": missing
        }
        
        return json.dumps(status, indent=2)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with the Barista Assistant
    await session.start(
        agent=BaristaAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))