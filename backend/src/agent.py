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

logger = logging.getLogger("day4_tutor")
load_dotenv(".env.local")

# Helper: find the content JSON file in common locations or via env var.
def find_content_path() -> Optional[str]:
    # 1) explicit env override
    env_path = os.environ.get("DAY4_CONTENT_PATH")
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
        os.path.join(os.getcwd(), "shared-data", "day4_tutor_content.json"),
        os.path.join(os.getcwd(), "backend", "day4_tutor_content.json"),
        os.path.join(os.path.dirname(__file__), "..", "shared-data", "day4_tutor_content.json"),
        os.path.join(os.path.dirname(__file__), "..", "day4_tutor_content.json"),
        os.path.join(os.path.dirname(__file__), "day4_tutor_content.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    return None


class Concept:
    def __init__(self, item: Dict):
        self.id = item.get("id")
        self.title = item.get("title")
        self.summary = item.get("summary")
        self.sample_question = item.get("sample_question")


class ContentStore:
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.concepts: List[Concept] = []
        if filepath:
            self._load(filepath)
        else:
            # no file: load default small content so the agent can still operate
            logger.warning("No content file provided; loading default sample content.")
            default = [
                {
                    "id": "variables",
                    "title": "Variables",
                    "summary": "Variables store values so you can reuse them later. They let you assign a name to a value so you can read or update it throughout your program.",
                    "sample_question": "What is a variable and why is it useful?"
                },
                {
                    "id": "loops",
                    "title": "Loops",
                    "summary": "Loops let you repeat an action multiple times without writing the same code again.",
                    "sample_question": "Explain the difference between a for loop and a while loop."
                }
            ]
            self.concepts = [Concept(item) for item in default]

    def _load(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.concepts = [Concept(item) for item in data]
            logger.info(f"Loaded {len(self.concepts)} concepts from content file: {path}")
        except Exception as e:
            logger.error(f"Error loading content file at {path}: {e}")
            # fall back to default content
            self.__init__(filepath=None)

    def list_concepts(self) -> List[Dict]:
        return [{"id": c.id, "title": c.title} for c in self.concepts]

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        if not concept_id:
            return None
        for c in self.concepts:
            if c.id == concept_id or (c.title and c.title.lower() == concept_id.lower()):
                return c
        return None

    def pick_random_concept(self) -> Optional[Concept]:
        return self.concepts[0] if self.concepts else None


class TutorAgent(Agent):
    def __init__(self, content_store: ContentStore):
        self.content = content_store
        self.mode: str = "learn"
        self.current_concept: Optional[Concept] = None
        self.history: List[Dict] = []
        super().__init__(
            instructions=(
                "You are Active Recall Coach — a warm, concise tutor that helps a learner by explaining, "
                "quizzing, and prompting them to teach back. Use the content provided to explain concepts, "
                "ask the sample question for quizzing, and in teach_back mode ask the learner to explain the "
                "concept back and give short, supportive qualitative feedback."
            )
        )

    def _log_interaction(self, kind: str, details: Dict):
        record = {
            "timestamp": datetime.now().isoformat(),
            "kind": kind,
            "mode": self.mode,
            "concept": self.current_concept.id if self.current_concept else None,
            "details": details,
        }
        self.history.append(record)
        logger.info(f"Recorded interaction: {record}")

    @function_tool
    async def list_concepts(self, context: RunContext):
        items = self.content.list_concepts()
        if not items:
            return "I don't have any concepts loaded yet."
        lines = [f"- {it['title']} (id: {it['id']})" for it in items]
        return "Here are the concepts I can help with:\n" + "\n".join(lines)

    @function_tool
    async def set_concept(self, context: RunContext, concept_id: str):
        c = self.content.get_concept(concept_id)
        if not c:
            return f"Sorry — I couldn't find a concept matching '{concept_id}'. You can say 'list concepts' to see available topics."
        self.current_concept = c
        self._log_interaction("select_concept", {"concept_id": c.id})
        return f"Great — we'll work on '{c.title}'. Which mode would you like? (learn / quiz / teach_back)"

    @function_tool
    async def set_mode(self, context: RunContext, mode: str):
        mode = (mode or "").strip().lower()
        if mode not in ("learn", "quiz", "teach_back"):
            return "I didn't recognize that mode. Please choose one of: learn, quiz, teach_back."
        prev = self.mode
        self.mode = mode
        self._log_interaction("switch_mode", {"from": prev, "to": mode})
        voice_map = {"learn": "Matthew", "quiz": "Alicia", "teach_back": "Ken"}
        return f"Switched to {mode} mode. I'll use the {voice_map.get(mode)} voice for this mode."

    @function_tool
    async def explain(self, context: RunContext, concept_id: Optional[str] = None):
        if concept_id:
            c = self.content.get_concept(concept_id)
            if not c:
                return f"Couldn't find '{concept_id}'. Try 'list concepts'."
            self.current_concept = c
        if not self.current_concept:
            return "Which concept would you like me to explain? You can say 'list concepts' to see options."
        c = self.current_concept
        self._log_interaction("explain", {"concept_id": c.id})
        return f"{c.title}: {c.summary}"

    @function_tool
    async def quiz(self, context: RunContext, concept_id: Optional[str] = None):
        if concept_id:
            c = self.content.get_concept(concept_id)
            if not c:
                return f"Couldn't find '{concept_id}'. Try 'list concepts'."
            self.current_concept = c
        if not self.current_concept:
            return "Which concept should I quiz you on? Try 'list concepts'."
        c = self.current_concept
        self._log_interaction("quiz_ask", {"concept_id": c.id})
        return f"Quiz — {c.sample_question}"

    @function_tool
    async def teach_back_prompt(self, context: RunContext, concept_id: Optional[str] = None):
        if concept_id:
            c = self.content.get_concept(concept_id)
            if not c:
                return f"Couldn't find '{concept_id}'. Try 'list concepts'."
            self.current_concept = c
        if not self.current_concept:
            return "Which concept should you teach back? Try 'list concepts'."
        c = self.current_concept
        self._log_interaction("teach_back_prompt", {"concept_id": c.id})
        return f"Please explain '{c.title}' back to me in your own words. Describe the main idea and give one short example."

    @function_tool
    async def evaluate_teach_back(self, context: RunContext, user_response: str):
        if not self.current_concept:
            return "I don't know which concept you just explained. Please set a concept first with 'set_concept'."
        c = self.current_concept
        keywords = set([w.strip(".,").lower() for w in c.summary.split() if len(w) > 3][:6])
        response_tokens = set([w.strip(".,").lower() for w in user_response.split()])
        matched = keywords.intersection(response_tokens)
        matched_count = len(matched)
        total = len(keywords) if len(keywords) > 0 else 1
        ratio = matched_count / total
        if ratio > 0.6:
            tone = "Great job — you covered the main points clearly."
            suggestion = "Try giving a short example or one step-by-step detail next time to make it even stronger."
        elif ratio > 0.3:
            tone = "Nice — you got several ideas across, though a few key terms were missing."
            suggestion = "You might try summarizing the core definition in one sentence before adding examples."
        else:
            tone = "Good effort — I can see pieces of the idea, but it would help to be a bit more focused."
            suggestion = "Try starting with a concise definition and then add one example."
        self._log_interaction("teach_back_eval", {"concept_id": c.id, "matched": list(matched), "score": ratio})
        return f"{tone} (keywords matched: {matched_count}/{total}). Suggestion: {suggestion}"

    @function_tool
    async def review_history(self, context: RunContext, count: int = 5):
        recent = self.history[-count:]
        lines = []
        for r in recent:
            lines.append(f"{r['timestamp']}: {r['kind']} on {r.get('concept')}")
        if not lines:
            return "No interactions recorded yet."
        return "Recent activity:\n" + "\n".join(lines)


# ----- Hooks used by the job process -----
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Determine content path and load content (with sensible fallbacks)
    content_path = find_content_path()
    if content_path:
        logger.info(f"Using content file: {content_path}")
        content = ContentStore(filepath=content_path)
    else:
        logger.error("Content file not found at expected locations. Falling back to built-in default content.")
        content = ContentStore(filepath=None)

    # Determine starting mode:
    mode_from_env = os.environ.get("DAY4_START_MODE")
    chosen_mode = None

    # If env var provided, prefer it (validate)
    if mode_from_env and mode_from_env.strip().lower() in ("learn", "quiz", "teach_back"):
        chosen_mode = mode_from_env.strip().lower()

    # If not provided, only prompt interactively when stdin is a TTY
    if not chosen_mode:
        if sys.stdin.isatty():
            try:
                print("\nWelcome to Day 4 — Teach-the-Tutor: Active Recall Coach")
                print("Available modes: learn (Matthew), quiz (Alicia), teach_back (Ken)")
                chosen_mode = input("Which mode would you like to start in? (learn/quiz/teach_back) ").strip().lower()
                if chosen_mode not in ("learn", "quiz", "teach_back"):
                    logger.warning("Invalid choice from prompt; defaulting to 'learn'.")
                    chosen_mode = "learn"
            except EOFError:
                logger.warning("Interactive input not available (EOF). Defaulting to 'learn' mode.")
                chosen_mode = "learn"
        else:
            # non-interactive environment (e.g., container, CI, or remote worker): choose default
            logger.info("Non-interactive environment detected; defaulting to 'learn' mode. To override, set DAY4_START_MODE env var.")
            chosen_mode = "learn"

    # Map chosen mode to the TTS voice (one-time selection at session start)
    voice_map = {"learn": "en-US-matthew", "quiz": "en-US-alicia", "teach_back": "en-US-ken"}
    selected_voice = voice_map.get(chosen_mode, "en-US-matthew")
    logger.info(f"Starting in '{chosen_mode}' mode with voice '{selected_voice}'")

    # Build session
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice=selected_voice,
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

    # Instantiate agent and set initial mode
    agent = TutorAgent(content_store=content)
    agent.mode = chosen_mode

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))