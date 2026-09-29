"""Vireo Support Intelligence — Configuration constants.

All business rules reference the support-policy.pdf section number.
Models are set via environment variables (OLLAMA_MODEL, OLLAMA_EMBED_MODEL).
"""

import os

# --- Ollama models (from `ollama list`) ---
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:latest")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# --- SLA targets (Policy §3) ---
# "Targets: chat 15 minutes, voice callback 2 hours, social 4 hours, email 8 hours."
SLA_TARGET_MINUTES = {
    "chat": 15,
    "voice": 120,
    "social": 240,
    "email": 480,
}

# --- SLA breach credit (Policy §3) ---
# "Every ticket that misses its first-response target automatically issues a store credit of Rs 350"
SLA_CREDIT_INR = 350

# --- Contact costs (Policy §4) ---
# "Fully loaded cost per contact: chat Rs 210, email Rs 260, voice callback Rs 520, social Rs 240"
CONTACT_COST_INR = {
    "chat": 210,
    "email": 260,
    "voice": 520,
    "social": 240,
}
BLENDED_CONTACT_COST_INR = 290  # Policy §4: "Blended across the current channel mix: Rs 290"

# --- Transfer cost (Policy §4) ---
# "Internal transfer between teams: Rs 305 per transfer"
TRANSFER_COST_INR = 305

# --- Replacement cost (Policy §5) ---
# "Replacement cost for planning: the product's unit cost plus Rs 340 for reverse pickup and forward shipping"
REPLACEMENT_SHIPPING_INR = 340

# --- Agent cost (Policy §4) ---
AGENT_COST_PER_HOUR = 165  # Policy §4: "Fully loaded agent cost: Rs 165 per agent-hour"
AGENT_SHIFT_HOURS = 8

# --- Goodwill cap (Policy §5) ---
GOODWILL_CAP_INR = 500  # Policy §5: "Goodwill credits are capped at Rs 500 per ticket"

# --- CSAT (Policy §8) ---
# "A one-question survey (1-5) is sent when a ticket is resolved"
CSAT_VALID_RANGE = (1, 5)

# --- Shifts (Policy §7) ---
# "Morning 06:00-14:00, Day 14:00-22:00, Night 22:00-06:00"
SHIFTS = {
    "Morning": (6, 14),
    "Day": (14, 22),
    "Night": (22, 6),  # wraps midnight
}

# --- Repeat contact window (Policy §10) ---
# "same customer does not contact again about the same issue within 30 days"
REPEAT_WINDOW_DAYS = 30

# --- Helpdesk cutover date (Policy §9) ---
# "The current helpdesk went live on 14 September 2025"
HELPDESK_CUTOVER = "2025-09-14"

# --- Status classifications ---
CLOSURE_STATUSES = {"resolved", "closed"}
ACTIVE_STATUSES = {"open", "pending"}

# --- Refund reason codes (Policy §5) ---
REFUND_REASON_CODES = {
    "GW-OTHER": "Goodwill / Other",
    "DOA-REPL": "Dead on arrival, refund chosen",
    "LOST-TRANSIT": "Lost or undelivered",
    "DUP-PAYMENT": "Duplicate or failed payment",
    "CANCEL": "Cancellation before dispatch",
    "PRICE-ADJ": "Price or coupon adjustment",
    "RETURN-QC-OK": "Return received and passed QC",
    "WTY-BUYBACK": "Warranty buy-back",
}

# --- Taxonomy (consolidated from Gemma discovery) ---
# Updated after running taxonomy discovery in classifier.py
TAXONOMY = [
    "Audio & Sound Quality",
    "Battery & Charging",
    "Connectivity & Bluetooth",
    "Delivery & Shipping",
    "App & Firmware Issues",
    "Returns & Refunds",
    "Billing & Payment",
    "Warranty & Repair",
    "Account & Login",
    "Product Enquiry",
    "Physical Damage & Build",
    "Other",
]

# --- Retrieval ---
RRF_K = 60  # Reciprocal Rank Fusion constant
DEFAULT_FEW_SHOT_K = 4

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")

# Ensure output directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)
