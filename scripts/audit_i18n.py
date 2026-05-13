"""Audit i18n JSON files for missing/untranslated strings."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

pt = json.load(open(ROOT / "src/i18n_strings/pt.json", encoding="utf-8"))
en = json.load(open(ROOT / "src/i18n_strings/en.json", encoding="utf-8"))
es = json.load(open(ROOT / "src/i18n_strings/es.json", encoding="utf-8"))


def flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, prefix + k + ".")
        else:
            yield prefix + k, v


pt_flat = dict(flat(pt))
en_flat = dict(flat(en))
es_flat = dict(flat(es))

# Keys in PT but not EN/ES
missing_en = set(pt_flat) - set(en_flat)
missing_es = set(pt_flat) - set(es_flat)
print(f"Missing from EN: {len(missing_en)}")
print(f"Missing from ES: {len(missing_es)}")

# Keys in code not in JSON
tr_pattern = re.compile(r"""tr\(['"]([^'"]+)['"]\)""")
found_keys = set()
for f in list(Path(ROOT / "src").rglob("*.py")) + [ROOT / "gui_app.py"]:
    try:
        src = open(f, encoding="utf-8").read()
        found_keys.update(tr_pattern.findall(src))
    except Exception:
        pass

# Only keys that look like dotted keys (section.key)
dotted = {k for k in found_keys if "." in k}
missing_code = sorted(dotted - set(pt_flat))
print(f"\nKeys used in code but missing from ALL JSON: {len(missing_code)}")
for k in missing_code:
    print(f"  {k}")

# EN strings that genuinely need translation (not acronyms/proper nouns/symbols)
skip_patterns = [
    r"^[A-Z][A-Z0-9 +×]+$",  # all-caps acronyms: EIS, DRT, DRT + EIS
    r"^[A-Z][a-z]+$",  # proper nouns: Nyquist, Ragone, Bode
    r"^[^a-zA-Z]*$",  # pure symbols: γ(τ) [Ω], τ (s)
    r"IonFlow",  # brand name
    r"^[-\w]+ \([^)]+\)$",  # units: Scan rate (A/g), CV (%)
    r"^n_taus$",  # technical identifier
    r"^Pipelines$",
    r"^Logs$",
    r"^Top 5$",
    r"^Radar$",
    r"^Box-plot$",
    r"Heatmap",
    r"^Reset DRT$",
]

import re as _re  # noqa: E402


def is_skip(s):
    return any(_re.search(p, s) for p in skip_patterns)


real_missing_en = [
    (k, v)
    for k, v in en_flat.items()
    if k in pt_flat and v == pt_flat[k] and not is_skip(v)
]

print(f"\nEN genuinely untranslated (non-trivial): {len(real_missing_en)}")
for k, v in sorted(real_missing_en):
    print(f"  {k}: {repr(v)}")

print("\n--- ES genuinely untranslated ---")
# ES: find strings matching PT that are clearly Portuguese (not Spanish cognates)
# Heuristic: if the PT string contains Portuguese-specific words
pt_markers = [
    "espec",
    "normalizado",
    "de Warburg",
    "típicos",
    "Sistemas",
    "menor",
    "Gráficos",
    "Circuito",
    "Ciclo",
    "Ajuste",
    "Número de Ciclos",
    "Fase",
    "Componente",
    "Valor",
    "Espectro",
]
real_missing_es = [
    (k, v)
    for k, v in es_flat.items()
    if k in pt_flat and v == pt_flat[k] and not is_skip(v)
]
print(f"Potentially untranslated ES strings: {len(real_missing_es)}")
for k, v in sorted(real_missing_es):
    print(f"  {k}: {repr(v)}")
