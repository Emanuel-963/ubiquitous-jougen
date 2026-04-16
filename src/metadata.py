import re


def extract_metadata(filename):
    name = filename.lower()

    if "li2so4" in name:
        electrolyte = "Li2SO4"
    elif "licl" in name:
        electrolyte = "LiCl"
    elif "h2so4" in name:
        electrolyte = "H2SO4"
    elif "na2so4" in name:
        electrolyte = "Na2SO4"
    elif "naoh" in name:
        electrolyte = "NaOH"
    else:
        electrolyte = "Unknown"

    if "0.1a" in name:
        current = "0.1A"
    elif "1a" in name:
        current = "1A"
    elif "10a" in name:
        current = "10A"
    else:
        current = "Unknown"

    if "gct" in name:
        treatment = "GCT"
    elif "gc" in name:
        treatment = "GC"
    elif "s316" in name:
        treatment = "Steel316"
    else:
        treatment = "None"

    return electrolyte, current, treatment


def extract_material_type(filename: str) -> str:
    """Extract material type (Nb2, Nb4, …) from filename.

    Looks for patterns like ``Nb2``, ``Nb4``, ``NF`` (nickel foam), etc.
    """
    name = filename.lower()
    # Match NbX pattern (Nb2, Nb4, Nb2O5, etc.)
    m = re.search(r'\bnb(\d)', name)
    if m:
        return f"Nb{m.group(1)}"
    if "nf" in name.split() or name.startswith("nf") or " nf " in f" {name} ":
        return "NF"
    return "Unknown"


def extract_synthesis_process(filename: str) -> str:
    """Extract synthesis process from filename.

    Detects 'Prisca' (or 'prisca', 'PRI') vs standard synthesis.
    Also detects alcohol-based (alcool/alcohol) and thermal (GCD) variants.
    """
    name = filename.lower()
    if "prisca" in name or "pri" in name.split():
        return "Prisca"
    if "alcool" in name or "alcohol" in name or "alc" in name.split():
        return "Alcohol"
    if "gcd" in name:
        return "GCD"
    return "Standard"


def extract_full_metadata(filename: str) -> dict:
    """Return a complete metadata dict for a sample filename.

    Combines legacy ``extract_metadata`` with new production variables.
    """
    electrolyte, current, treatment = extract_metadata(filename)
    return {
        "Electrolyte": electrolyte,
        "Current": current,
        "Treatment": treatment,
        "Material_Type": extract_material_type(filename),
        "Synthesis": extract_synthesis_process(filename),
    }
