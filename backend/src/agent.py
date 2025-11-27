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

logger = logging.getLogger("fraud_agent")
load_dotenv(".env.local")

# --- Fraud Database Handling ---
class FraudCaseDB:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.cases: List[Dict] = []
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.cases = json.load(f)
                logger.info(f"Loaded {len(self.cases)} fraud cases from {self.filepath}")
            else:
                logger.warning(f"Fraud DB file not found at {self.filepath}")
                self.cases = []
        except Exception as e:
            logger.error(f"Error loading fraud DB at {self.filepath}: {e}")
            self.cases = []

    def get_case_by_username(self, username: str) -> Optional[Dict]:
        for case in self.cases:
            if case["userName"].lower() == username.lower():
                return case
        return None

    def update_case(self, username: str, status: str, note: str):
        for case in self.cases:
            if case["userName"].lower() == username.lower():
                case["status"] = status
                case["outcome_note"] = note
                self._save()
                return True
        return False

    def _save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.cases, f, indent=2)
            logger.info(f"Saved fraud DB to {self.filepath}")
        except Exception as e:
            logger.error(f"Error saving fraud DB: {e}")

# --- Fraud Agent ---
class FraudAgent(Agent):
    def __init__(self, db: FraudCaseDB):
        super().__init__(
            instructions=(
                "You are a Fraud Detection Representative for Bank of India. "
                "Your goal is to verify a suspicious transaction with the customer. "
                "1. Start by saying 'Hello, this is a Fraud Detection Representative for Bank of India' and state the reason for the call (suspicious transaction). "
                "2. Ask for the customer's name to look up their file. "
                "3. Once you have the name, verify their identity using their security question. "
                "4. If verified, read out the transaction details (Merchant, Amount, Time, Location). "
                "5. Ask if they authorized this transaction. "
                "6. If YES: Mark as safe, thank them, and end call. "
                "7. If NO: Mark as fraud, explain that the card is blocked and a new one is on the way, then end call. "
                "8. If verification fails or user is unknown: Apologize and end call. "
                "Be professional, calm, and reassuring. Do NOT ask for real card numbers or passwords."
            )
        )
        self.db = db
        self.current_case: Optional[Dict] = None
        self.verified = False

    @function_tool
    async def lookup_user(self, context: RunContext, name: str):
        """Look up a customer by name to find their fraud case."""
        case = self.db.get_case_by_username(name)
        if case:
            self.current_case = case
            return f"I found a case for {name}. Please ask them the security question: {case['securityQuestion']}"
        else:
            return "I could not find a customer with that name."

    @function_tool
    async def verify_security_answer(self, context: RunContext, answer: str):
        """Check if the user's answer to the security question is correct."""
        if not self.current_case:
            return "No user loaded."
        
        expected = self.current_case["securityAnswer"].lower()
        if answer.lower() == expected:
            self.verified = True
            c = self.current_case
            details = f"Transaction at {c['transactionName']} for {c['transactionAmount']} on {c['transactionTime']} in {c['transactionLocation']}."
            return f"Identity verified. Here are the transaction details: {details}. Ask if they authorized it."
        else:
            return "Security answer incorrect."

    @function_tool
    async def process_transaction_response(self, context: RunContext, authorized: bool):
        """Process the user's confirmation or denial of the transaction."""
        if not self.current_case or not self.verified:
            return "Cannot process without verified user."
        
        username = self.current_case["userName"]
        if authorized:
            self.db.update_case(username, "confirmed_safe", "Customer confirmed transaction.")
            return "Marked as safe. You can end the call."
        else:
            self.db.update_case(username, "confirmed_fraud", "Customer denied transaction. Card blocked.")
            return "Marked as fraud. Inform user card is blocked and end call."

# ----- Hooks used by the job process -----
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Initialize DB
    # Look for fraud_db.json in the same directory or src
    db_path = os.path.join(os.path.dirname(__file__), "fraud_db.json")
    if not os.path.exists(db_path):
         db_path = os.path.join(os.getcwd(), "src", "fraud_db.json")
    
    db = FraudCaseDB(db_path)

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
    agent = FraudAgent(db=db)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))