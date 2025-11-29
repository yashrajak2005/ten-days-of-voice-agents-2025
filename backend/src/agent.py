import logging
import random
import json
from typing import Optional

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
    llm,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit import rtc

logger = logging.getLogger("gamemaster_agent")
load_dotenv(".env.local")

class GameMasterAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are the Dungeon Master (GM) for a fantasy tabletop RPG set in the Kingdom of Eldoria. "
                "Your goal is to guide the player through an immersive interactive story. "
                "1. **Tone**: Atmospheric, descriptive, and engaging. Use sound effects descriptions in your speech if appropriate (e.g., *creak*). "
                "2. **Role**: You describe the scene, the NPCs, and the consequences of the player's actions. "
                "3. **Interaction Loop**: "
                "   - Describe the current situation or scene vividly. "
                "   - End your turn by asking the player 'What do you do?' or a similar prompt. "
                "4. **Mechanics**: "
                "   - If the player attempts an action with a chance of failure (e.g., attacking, sneaking, persuading), use the 'roll_dice' tool. "
                "   - Narrate the outcome based on the roll result (High = Success, Low = Failure). "
                "5. **Memory**: Keep track of the story progression, key items found, and NPCs met based on the conversation history. "
                "6. **Start**: Begin the session by setting the scene. The player starts in a mysterious location (e.g., a dark forest, a bustling tavern, or a dungeon cell). "
                "7. Be creative and adapt to whatever the player says. "
            )
        )

    @function_tool
    async def roll_dice(self, context: RunContext, sides: int = 20, count: int = 1):
        """Rolls virtual dice for checks or damage. Default is 1d20."""
        rolls = [random.randint(1, sides) for _ in range(count)]
        total = sum(rolls)
        result_str = ", ".join(map(str, rolls))
        return f"Rolled {count}d{sides}: {result_str} (Total: {total})"

# ----- Hooks used by the job process -----
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

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
    agent = GameMasterAgent()

    # Hook to send chat messages when the agent speaks
    @session.on("response_done")
    def _on_response_done(response: llm.LLMStream):
        # The conversation context should now have the assistant's response appended.
        # We'll retrieve the last message and send it to the chat.
        try:
            if session.chat_ctx.messages:
                last_msg = session.chat_ctx.messages[-1]
                if last_msg.role == llm.ChatRole.ASSISTANT and last_msg.content:
                    # Send the text to the room so the frontend can display it
                    # We need to run this async, so we use asyncio.create_task
                    import asyncio
                    import time
                    
                    # The content might be a string or a list of ContentItems.
                    text_content = ""
                    if isinstance(last_msg.content, str):
                        text_content = last_msg.content
                    elif isinstance(last_msg.content, list):
                        text_content = " ".join([c if isinstance(c, str) else "" for c in last_msg.content])
                    
                    if text_content:
                        # Construct the payload for LiveKit Chat
                        # The standard topic is "lk-chat-topic"
                        payload = json.dumps({
                            "message": text_content,
                            "timestamp": int(time.time() * 1000)
                        }).encode("utf-8")
                        
                        asyncio.create_task(ctx.room.local_participant.publish_data(
                            payload=payload,
                            topic="lk-chat-topic"
                        ))
        except Exception as e:
            logger.error(f"Error sending chat message: {e}")

    @ctx.room.on("data_received")
    def _on_data_received(data: rtc.DataPacket):
        try:
            msg = data.data.decode("utf-8")
            if msg == "restart":
                logger.info("Restart command received. Resetting session.")
                # To restart, we can clear the chat context and trigger a new greeting.
                session.chat_ctx.messages.clear()
                
                # Create a new task to handle the restart logic async
                async def restart_sequence():
                     # Add a user message to prompt the restart
                    fake_user_msg = llm.ChatMessage(
                        role=llm.ChatRole.USER,
                        content="Restart the story. Forget everything and start from the beginning. Greet me as if we just met."
                    )
                    # We can't directly inject into the running loop easily without a trigger.
                    # But we can use the conversation context.
                    session.chat_ctx.messages.append(fake_user_msg)
                    await session.response.create()
                
                # Schedule the restart
                import asyncio
                asyncio.create_task(restart_sequence())

        except Exception as e:
            logger.error(f"Error handling data message: {e}")

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))