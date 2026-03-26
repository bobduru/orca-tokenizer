# ============================================================
# Vocabulary Definition
# ============================================================
# Modify these sets to update the allowed tokens

VOCAB = {
    # Pulse Repetition Rate (PRR)
    "prr": {"VERTICAL", "BLUR", "TIGHT", "NORMAL", "SPACED"},
    
    # Contour
    "contour": {"NONE","UP", "DOWN", "FLAT", "PEAK", "VALLEY", "SQUIGGLE"},
    
    # Modifiers (speed / duration)
    "modifier": {"SLOW", "FAST", "SHORT", "LEFT", "RIGHT", "NOISY", "T",},
    
    # Special tokens
    "special": {"GAP", },
}

# Flatten all valid tokens into one set
ALL_VALID_TOKENS = set()
for category, tokens in VOCAB.items():
    ALL_VALID_TOKENS |= tokens

# ============================================================
# Combinatoric rules
# ============================================================

# PRRs that support contours (others are contour-less units)
PRR_HAS_CONTOUR = {"TIGHT", "NORMAL", "SPACED"}

# Modifier → which contours it can apply to
MODIFIER_APPLIES_TO = {
    "SLOW":  {"UP", "DOWN", "PEAK", "VALLEY"},
    "FAST":  {"UP", "DOWN", "PEAK", "VALLEY"},
    "SHORT": {"UP", "DOWN", "FLAT", "PEAK", "VALLEY", "NONE"},
    "LEFT":  {"PEAK", "VALLEY"},
    "RIGHT": {"PEAK", "VALLEY"},
    "UP":    {"SQUIGGLE"},
    "DOWN":  {"SQUIGGLE"},
    "NOISY": {"UP", "DOWN", "FLAT", "PEAK", "VALLEY", "SQUIGGLE", "NONE"},
    "T":     {"UP", "DOWN",},
}

# Allowed punctuation within label content (not treated as tokens)
ALLOWED_PUNCTUATION = {"?", "|"}

# ============================================================
# Syllable generation (PRR_contour_modifiers)
# ============================================================
# Mutually exclusive modifier groups: at most one per group
MODIFIER_MUTEX_GROUPS = [
    {"SLOW", "FAST"},
    {"LEFT", "RIGHT"},
    {"UP", "DOWN"},  # as trend modifiers for SQUIGGLE only
]
# Independent modifiers (each 0 or 1)
MODIFIER_INDEPENDENT = {"SHORT", "NOISY", "T"}


def _iter_modifier_subsets(contour: str, remove_modifiers: list):
    """Yield all valid modifier subsets for a given contour (each as sorted tuple)."""
    valid = {m for m, contours in MODIFIER_APPLIES_TO.items() if contour in contours}
    if remove_modifiers:
        remove = set(remove_modifiers)
        valid = valid - remove
    if not valid:
        yield ()
        return

    mutex = [[m for m in g if m in valid] for g in MODIFIER_MUTEX_GROUPS]
    indep = sorted(m for m in MODIFIER_INDEPENDENT if m in valid)

    def _recurse(i_mutex: int, i_indep: int, chosen: list):
        if i_mutex == len(mutex):
            if i_indep == len(indep):
                yield tuple(sorted(chosen))
                return
            yield from _recurse(i_mutex, i_indep + 1, chosen)
            yield from _recurse(i_mutex, i_indep + 1, chosen + [indep[i_indep]])
            return
        group = mutex[i_mutex]
        if not group:
            yield from _recurse(i_mutex + 1, i_indep, chosen)
            return
        yield from _recurse(i_mutex + 1, i_indep, chosen)
        for m in group:
            yield from _recurse(i_mutex + 1, i_indep, chosen + [m])

    yield from _recurse(0, 0, [])


def generate_all_syllables(remove_modifiers: list | None = None):
    """Return list of all possible syllable strings (PRR_contour_modifiers, GAP included)."""
    remove_modifiers = remove_modifiers or []
    out = ["GAP"]
    for prr in VOCAB["prr"]:
        if prr not in PRR_HAS_CONTOUR:
            for mods in _iter_modifier_subsets("NONE", remove_modifiers):
                out.append("_".join([prr, "NONE"] + list(mods)))
    contours_full = ("UP", "DOWN", "FLAT", "PEAK", "VALLEY", "SQUIGGLE")
    for prr in PRR_HAS_CONTOUR:
        for contour in contours_full:
            for mods in _iter_modifier_subsets(contour, remove_modifiers):
                out.append("_".join([prr, contour] + list(mods)))
    return out


