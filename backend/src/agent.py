import logging
import json
import os
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
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class CheckInState:
    def __init__(self):
        self.date: str = datetime.now().strftime("%Y-%m-%d")
        self.mood: Optional[str] = None
        self.energy_level: Optional[str] = None
        self.stress_factors: List[str] = []
        self.daily_objectives: List[str] = []
        self.additional_notes: List[str] = []
        self.agent_reflections: List[str] = []
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date,
            "mood": self.mood,
            "energy_level": self.energy_level,
            "stress_factors": self.stress_factors,
            "daily_objectives": self.daily_objectives,
            "additional_notes": self.additional_notes,
            "agent_reflections": self.agent_reflections,
            "timestamp": datetime.now().isoformat()
        }
    
    def is_complete(self) -> bool:
        return all([
            self.mood is not None,
            self.energy_level is not None,
            len(self.daily_objectives) > 0
        ])
    
    def get_missing_fields(self) -> List[str]:
        missing = []
        if not self.mood:
            missing.append("how you're feeling today")
        if not self.energy_level:
            missing.append("your energy level")
        if not self.daily_objectives:
            missing.append("your objectives or intentions for today")
        return missing


class WellnessLog:
    def __init__(self, filepath: str = "wellness_log.json"):
        self.filepath = filepath
        self.history: List[Dict] = []
        self.load_history()
    
    def load_history(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.history = data.get("check_ins", [])
                logger.info(f"Loaded {len(self.history)} previous check-ins")
            except Exception as e:
                logger.error(f"Error loading wellness log: {e}")
                self.history = []
        else:
            logger.info("No existing wellness log found, starting fresh")
            self.history = []
    
    def save_check_in(self, check_in_data: Dict):
        self.history.append(check_in_data)
        
        data = {
            "check_ins": self.history,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved check-in to {self.filepath}")
        except Exception as e:
            logger.error(f"Error saving wellness log: {e}")
    
    def get_last_check_in(self) -> Optional[Dict]:
        if self.history:
            return self.history[-1]
        return None
    
    def get_recent_check_ins(self, count: int = 3) -> List[Dict]:
        return self.history[-count:] if len(self.history) >= count else self.history
    
    def get_summary_for_context(self) -> str:
        if not self.history:
            return "This is your first check-in with me. I'm looking forward to getting to know you!"
        
        last_check_in = self.get_last_check_in()
        
        summary_parts = []
        summary_parts.append(f"Last time we talked on {last_check_in['date']}")
        
        if last_check_in.get('mood'):
            summary_parts.append(f"you mentioned feeling {last_check_in['mood']}")
        
        if last_check_in.get('energy_level'):
            summary_parts.append(f"with {last_check_in['energy_level']} energy")
        
        if last_check_in.get('daily_objectives'):
            objectives = last_check_in['daily_objectives']
            if len(objectives) > 0:
                summary_parts.append(f"and you wanted to work on: {', '.join(objectives[:2])}")
        
        return ". ".join(summary_parts) + "."


class Assistant(Agent):
    def __init__(self) -> None:
        self.current_check_in = CheckInState()
        self.wellness_log = WellnessLog()
        
        previous_context = self.wellness_log.get_summary_for_context()
        
        super().__init__(
            instructions=f"""You are Wellness Companion, a supportive and grounded daily health and wellness voice assistant.
            Your role is to conduct brief, meaningful check-ins that help users reflect on their well-being and set intentions for their day.
            
            Your personality:
            - Warm, empathetic, and non-judgmental
            - Grounded and realistic (never make medical claims or diagnoses)
            - Encouraging but honest
            - A good listener who asks thoughtful follow-up questions
            - Brief and focused - keep the conversation productive but not overwhelming
            
            Your approach to check-ins:
            1. Start with a warm greeting and ask about their mood and how they're feeling today
            2. Ask about their energy level and any stress factors
            3. Explore their intentions and objectives for the day (1-3 things they'd like to accomplish)
            4. Offer simple, practical, and actionable reflections or suggestions when appropriate
            5. End with a brief recap to confirm understanding
            
            When offering advice or reflections:
            - Keep suggestions small, practical, and achievable
            - Focus on actionable steps (e.g., "try taking a 5-minute walk," "break that task into smaller steps")
            - Encourage self-care basics: rest, movement, breaks, connection with others
            - NEVER diagnose, prescribe, or offer medical advice
            - If someone mentions serious mental or physical health concerns, gently encourage them to speak with a healthcare professional
            
            Context from previous check-ins:
            {previous_context}
            
            Use this context naturally in conversation to show continuity and care. For example:
            - "Last time you mentioned feeling low on energy. How does today compare?"
            - "You wanted to focus on that project yesterday. How did that go?"
            
            Keep the conversation natural, supportive, and focused on helping them have a better day.
            Don't use complex formatting in your speech - just talk naturally as a supportive companion.""",
        )

    @function_tool
    async def record_mood(self, context: RunContext, mood_description: str):
        logger.info(f"Recording mood: {mood_description}")
        self.current_check_in.mood = mood_description
        return f"I've noted that you're feeling {mood_description} today."

    @function_tool
    async def record_energy_level(self, context: RunContext, energy_description: str):
        logger.info(f"Recording energy level: {energy_description}")
        self.current_check_in.energy_level = energy_description
        return f"Got it, your energy is {energy_description} today."

    @function_tool
    async def add_stress_factor(self, context: RunContext, stress_description: str):
        logger.info(f"Adding stress factor: {stress_description}")
        if stress_description not in self.current_check_in.stress_factors:
            self.current_check_in.stress_factors.append(stress_description)
        return f"I understand that {stress_description} is weighing on you."

    @function_tool
    async def add_daily_objective(self, context: RunContext, objective: str):
        logger.info(f"Adding objective: {objective}")
        if objective not in self.current_check_in.daily_objectives:
            self.current_check_in.daily_objectives.append(objective)
        return f"Great, I've added '{objective}' to your intentions for today."

    @function_tool
    async def add_reflection(self, context: RunContext, reflection: str):
        logger.info(f"Adding reflection: {reflection}")
        self.current_check_in.agent_reflections.append(reflection)
        return "I'm glad I could offer that perspective."

    @function_tool
    async def add_note(self, context: RunContext, note: str):
        logger.info(f"Adding note: {note}")
        self.current_check_in.additional_notes.append(note)
        return "I've noted that."

    @function_tool
    async def check_progress(self, context: RunContext):
        logger.info("Checking check-in progress")
        missing = self.current_check_in.get_missing_fields()
        
        if not missing:
            return "We've covered the key areas! Let me give you a quick recap to make sure I understood everything correctly."
        else:
            missing_str = ", ".join(missing)
            return f"I'd still like to hear about {missing_str} before we wrap up."

    @function_tool
    async def complete_check_in(self, context: RunContext):
        logger.info("Attempting to complete check-in")
        
        if not self.current_check_in.is_complete():
            missing = self.current_check_in.get_missing_fields()
            missing_str = ", ".join(missing)
            return f"Before we finish, I'd like to hear about {missing_str}."
        
        check_in_data = self.current_check_in.to_dict()
        self.wellness_log.save_check_in(check_in_data)
        
        mood = self.current_check_in.mood
        objectives = self.current_check_in.daily_objectives
        objectives_text = ", ".join(objectives) if len(objectives) <= 3 else ", ".join(objectives[:3]) + f" and {len(objectives) - 3} more"
        
        closing = f"Perfect! I've saved today's check-in. You're feeling {mood}, and you're focusing on: {objectives_text}. "
        
        if len(self.current_check_in.agent_reflections) > 0:
            closing += "Remember the small steps we discussed. "
        
        closing += "I'm here whenever you need to check in. Take care, and have a good day!"
        
        logger.info(f"Check-in completed and saved for {self.current_check_in.date}")
        
        self.current_check_in = CheckInState()
        
        return closing

    @function_tool
    async def review_previous_check_ins(self, context: RunContext, days: int = 3):
        logger.info(f"Reviewing previous {days} check-ins")
        recent = self.wellness_log.get_recent_check_ins(days)
        
        if not recent:
            return "I don't have any previous check-ins recorded yet."
        
        summary = f"Looking at your last {len(recent)} check-in(s): "
        summaries = []
        
        for check_in in recent:
            date = check_in.get('date', 'unknown date')
            mood = check_in.get('mood', 'not specified')
            energy = check_in.get('energy_level', 'not specified')
            summaries.append(f"On {date}, you felt {mood} with {energy} energy")
        
        summary += ". ".join(summaries) + "."
        return summary


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
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

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))