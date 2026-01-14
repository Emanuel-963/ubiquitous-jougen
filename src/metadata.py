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
