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

logger = logging.getLogger("sdr_agent")
load_dotenv(".env.local")

# Helper: find the content JSON file in common locations or via env var.
def find_content_path() -> Optional[str]:
    # 1) explicit env override
    env_path = os.environ.get("RAZORPAY_CONTENT_PATH")
    if env_path:
        if os.path.isabs(env_path):
            candidate = env_path
        else:
            candidate = os.path.join(os.getcwd(), env_path)
        if os.path.exists(candidate):
            return candidate
        # try relative to project
        alt = os.path.join(os.path.dirname(__file__), "..", env_path)
        if os.path.exists(alt):
            return os.path.abspath(alt)

    # 2) common locations (project-root/shared-data, backend/, repo layout)
    candidates = [
        os.path.join(os.path.dirname(__file__), "razorpay_data.json"),
        os.path.join(os.getcwd(), "backend", "src", "razorpay_data.json"),
        os.path.join(os.getcwd(), "src", "razorpay_data.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    return None


class CompanyInfo:
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.data: Dict = {}
        if filepath:
            self._load(filepath)
        else:
            logger.warning("No content file provided; loading empty data.")

    def _load(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            logger.info(f"Loaded company info from: {path}")
        except Exception as e:
            logger.error(f"Error loading content file at {path}: {e}")

    def get_faqs(self) -> List[Dict]:
        return self.data.get("faqs", [])

    def get_pricing(self) -> Dict:
        return self.data.get("pricing", {})

    def get_description(self) -> str:
        return self.data.get("description", "")


class SDRAgent(Agent):
    def __init__(self, company_info: CompanyInfo):
        self.company_info = company_info
        self.lead_info: Dict = {}
        super().__init__(
            instructions=(
                "You are a Sales Development Representative (SDR) for Razorpay, a leading fintech company in India. "
                "Your goal is to answer user questions about Razorpay politely and professionally, and then collect lead information. "
                "You should greet the user warmly, ask what brought them here, and answer their questions using the provided tools. "
                "Do NOT make up information. If you don't know, say you'll check with a specialist. "
                "After answering their initial questions, naturally transition to asking for their details: Name, Company, Email, Role, Use Case, Team Size, and Timeline. "
                "Don't ask for everything at once; make it conversational. "
                "When the user indicates they are done (e.g., 'That's all', 'Thanks'), use the 'end_call_summary' tool to summarize and save the lead."
            )
        )

    @function_tool
    async def answer_question(self, context: RunContext, query: str):
        """Search the FAQ and company info to answer a user's question about product, pricing, or company."""
        query = query.lower()
        faqs = self.company_info.get_faqs()
        
        # Simple keyword matching
        best_match = None
        max_score = 0
        
        for faq in faqs:
            q_tokens = set(faq["question"].lower().split())
            query_tokens = set(query.split())
            overlap = len(q_tokens.intersection(query_tokens))
            if overlap > max_score:
                max_score = overlap
                best_match = faq["answer"]

        if "price" in query or "cost" in query or "fee" in query:
             pricing = self.company_info.get_pricing()
             return f"Here is our pricing structure: Standard plan is {pricing.get('standard_plan', {}).get('platform_fee', '2%')}. We also have an Enterprise plan for larger volumes."

        if best_match and max_score > 0:
            return best_match
        
        return "I'm not exactly sure about that specific detail. I can connect you with a product specialist for a deeper dive. Is there anything else I can help with regarding our general offerings?"

    @function_tool
    async def collect_lead_info(self, context: RunContext, field: str, value: str):
        """Save a specific piece of lead information (Name, Company, Email, Role, Use Case, Team Size, Timeline)."""
        self.lead_info[field] = value
        logger.info(f"Collected lead info: {field} = {value}")
        return f"Got it, saved {field}."

    @function_tool
    async def end_call_summary(self, context: RunContext):
        """Call this when the user says they are done to generate a summary and save the lead."""
        summary_text = (
            f"Thanks for chatting! Here's a quick summary: You are {self.lead_info.get('Name', 'a visitor')} "
            f"from {self.lead_info.get('Company', 'a company')}, looking to use Razorpay for {self.lead_info.get('Use Case', 'payments')}. "
            f"We'll be in touch shortly at {self.lead_info.get('Email', 'your email')}."
        )
        
        # Save to file
        output_file = "lead_summary.json"
        try:
            with open(output_file, "w") as f:
                json.dump(self.lead_info, f, indent=2)
            logger.info(f"Saved lead summary to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save lead summary: {e}")

        return summary_text


# ----- Hooks used by the job process -----
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Determine content path and load content
    content_path = find_content_path()
    if content_path:
        logger.info(f"Using content file: {content_path}")
        company_info = CompanyInfo(filepath=content_path)
    else:
        logger.error("Content file not found. Using empty company info.")
        company_info = CompanyInfo(filepath=None)

    # Build session
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", # Keep Matthew as the default professional voice
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
    agent = SDRAgent(company_info=company_info)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))