# PalmSens — Partnership Outreach Email Template

**To:** developer@palmsens.com  (or: applications@palmsens.com)  
**CC:** *(your institutional email)*  
**Subject:** Open-Source EIS Analysis Software — Integration/Partnership Proposal

---

Dear PalmSens team,

My name is **[YOUR NAME]**, a researcher at **[YOUR INSTITUTION]** working on
electrochemical characterisation of energy-storage materials.

Over the past year I have developed **IonFlow Pipeline**, an open-source Python
desktop tool for comprehensive EIS data analysis.  I am writing to explore
whether a technical collaboration or bundling arrangement could benefit both
projects.

## What IonFlow Pipeline Offers

- **Multi-format parser** — already reads BioLogic `.mpt`/`.mpr`, Gamry `.dta`,
  Zahner `.idf`, and generic CSV.  A PalmSens `.pssession` / `.csv` parser
  would be a natural addition.
- **Automated circuit fitting** — 11 equivalent circuits with physics-based
  initial estimates and an ML-assisted model selector.
- **DRT analysis** — Tikhonov-regularised distribution of relaxation times,
  complementary to PalmSens' Methodica EIS add-on.
- **Kramers-Kronig validation** — lin-KK residual check before every fit.
- **Branded PDF reports** — white-label-ready: logo, author, institution fields.
- **Python API** — fully scriptable; easy to embed in PSTrace or PStouch.
- **2 056 passing tests** (pytest, Python 3.13) — production-quality codebase.
- **Open source (MIT)** — no licensing friction for customers.

## Concrete Collaboration Ideas

1. **Native PalmSens parser** — I add a `.pssession` / CSV reader; PalmSens
   acknowledges IonFlow Pipeline in documentation or application notes.

2. **Application Note / Bundling** — PalmSens includes IonFlow Pipeline
   as a recommended analysis companion for EmStat Pico / PalmSens4 users
   (similar to how Gamry recommends Echem Analyst).

3. **OEM SDK integration** — IonFlow's `ionflow-sdk` Python package embedded
   directly in PSTrace or a web dashboard, providing DRT + advanced fitting
   to PalmSens customers who need more than impedance.py.

4. **Co-authored application note** — we demonstrate a full workflow
   (PalmSens measurement → IonFlow analysis → branded PDF) for solid-state
   batteries or corrosion monitoring.

## Evidence of Quality

- GitHub: <https://github.com/Emanuel-963/ubiquitous-jougen>
- Latest release: v0.3.1 (Windows installer + pip package)
- JOSS paper draft submitted / in preparation

I would be happy to arrange a 30-minute video call to demonstrate the tool
and discuss what value we could create together.

Thank you for your time.

Best regards,  
**[YOUR NAME]**  
[YOUR TITLE] | [YOUR INSTITUTION]  
[YOUR EMAIL] | [YOUR PHONE]  
ORCID: [0000-0000-0000-0000]

---

## Alternative: Gamry Instruments

**To:** support@gamry.com  
**Subject:** IonFlow Pipeline — Open-Source EIS Analysis, Gamry .dta Native Support

*(Same structure as above; emphasise that Gamry .dta parsing already works
natively and offer to add EXPLAIN/MIXED sequence file support.)*

---

## Alternative: Metrohm Autolab

**To:** autolab@metrohm.com  
**Subject:** Collaboration Proposal — Open-Source EIS Post-Processing Tool

*(Emphasise NOVA `.nox` / `.csv` export support, Nova scripting bridge via
Python API, and the fact that IonFlow can complement NOVA's built-in analysis
by providing DRT, ML circuit selection, and batch PDF reports.)*

---

## Alternative: Zahner

**To:** zahner@zahner.de  
**Subject:** IonFlow Pipeline — Native Zahner .idf Parser, Potential Collaboration

*(Zahner .idf already supported.  Offer Thales XT integration,
EPC42 / Zennium workflow documentation.)*

---

## Follow-up Schedule

| Day  | Action |
|------|--------|
|  0   | Send email to PalmSens |
|  7   | Follow-up if no reply |
|  14  | Send to Gamry if PalmSens declines |
|  21  | Send to Metrohm / Zahner |
|  30  | Evaluate responses, schedule demos |
