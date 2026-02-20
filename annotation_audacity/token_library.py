# ============================================================
# Vocabulary Definition
# ============================================================
# Modify these sets to update the allowed tokens

VOCAB = {
    # Pulse Repetition Rate (PRR)
    "prr": {"VERTICAL", "BLUR", "TIGHT", "NORMAL", "SPACED"},
    
    # Contour
    "contour": {"UP", "DOWN", "FLAT", "PEAK", "VALLEY", "SQUIGGLE"},
    
    # Modifiers (speed / duration)
    "modifier": {"SLOW", "FAST", "SHORT", "LEFT", "RIGHT"},
    
    # Special tokens
    "special": {"GAP", "NOISY", "T"},
}

# Flatten all valid tokens into one set
ALL_VALID_TOKENS = set()
for category, tokens in VOCAB.items():
    ALL_VALID_TOKENS |= tokens

# Allowed punctuation within label content (not treated as tokens)
ALLOWED_PUNCTUATION = {"?", "|"}
