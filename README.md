# MediEdge — Offline Clinical Intelligence

> AI-powered clinical decision support for community health workers in resource-limited settings. Powered by Gemma. Runs completely offline on any laptop or mobile phone.

---

## The Problem

4.5 billion people lack access to a specialist physician within their community. In rural Nigeria, a health worker may be the only medical presence for 200 kilometres. When connectivity fails and specialists are unreachable — patients die from conditions that are diagnosable and treatable.

MediEdge was built to close that gap.

---

## The Solution

MediEdge is a multimodal offline clinical decision support system that accepts voice, image, and text input and produces structured clinical assessments including:

- Triage severity — RED (immediate), AMBER (within 4h), GREEN (routine)
- Differential diagnoses ranked by likelihood with ICD-10 codes
- First-line treatment protocols sourced from WHO Essential Medicines guidelines
- Specific drug dosages and routes of administration
- Drug interaction warnings and contraindication alerts
- Referral flags with urgency level and appropriate facility type
- Downloadable PDF clinical report for paper-based referral handoff

**No internet. No cloud. No data leaves the device.**

---

## Architecture

```
PWA (Mobile + Desktop — installable, offline-capable)
              ↓
FastAPI Backend (localhost:8000)
              ↓
Ollama → gemma3:4b (local inference, no internet)
ChromaDB → WHO/CDC Clinical Knowledge Base (RAG)
ReportLab → PDF Report Generation
```

---

## Quick Start

### Requirements

- Python 3.12+
- [Ollama](https://ollama.ai) installed
- Model downloaded: `ollama pull gemma3:4b`

### Step 1 — Install backend dependencies

```bash
cd backend
pip install fastapi uvicorn httpx chromadb reportlab python-multipart
```

### Step 2 — Start the backend

```bash
python main.py
```

Backend runs at `http://localhost:8000`

### Step 3 — Start the frontend

Open a second terminal:

```bash
cd pwa/public
python -m http.server 5500
```

### Step 4 — Open the app

Open Chrome and go to:

```
http://localhost:5500
```

### Mobile access

1. Connect your phone to the same WiFi network as your laptop
2. Find your laptop IP: run `ipconfig` and look for IPv4 Address
3. On your phone browser open: `http://YOUR_LAPTOP_IP:5500`
4. Go to Settings in the app and set Backend URL to: `http://YOUR_LAPTOP_IP:8000`
5. Tap Save & reconnect — the green dot confirms connection

---

## Clinical Knowledge Base
MediEdge contains a built-in WHO/CDC clinical knowledge base 
covering 15 high-burden conditions with complete treatment protocols:

| Condition                              | ICD-10  | Triage |
|----------------------------------------|---------|--------|
| Plasmodium falciparum malaria (severe) | B50.9   | RED    |
| Bacterial meningitis                   | A39.0   | RED    |
| Pre-eclampsia with severe features     | O14.1   | RED    |
| STEMI                                  | I21.0   | RED    |
| Sepsis                                 | A41.9   | RED    |
| Severe acute asthma                    | J46     | RED    |
| Acute ischaemic stroke                 | I63.9   | RED    |
| Major thermal burns                    | T31.2   | RED    |
| Diabetic ketoacidosis                  | E11.10  | RED    |
| Anaphylaxis                            | T78.2   | RED    |
| Community-acquired pneumonia           | J18.9   | RED    |
| Acute gastroenteritis with dehydration | A09     | AMBER  |
| Typhoid fever                          | A01.0   | AMBER  |
| Pulmonary tuberculosis                 | A15.0   | AMBER  |
| Urinary tract infection                | N39.0   | AMBER  |

Each condition includes immediate actions, investigations, treatment 
with WHO Essential Medicine drug dosages, monitoring parameters, 
referral criteria, drug alerts, and red flag warning signs — all 
sourced from WHO and CDC guidelines.

---

## Clinical Validation

MediEdge was tested against high-burden clinical scenarios:

**Severe malaria:** 8-year-old, 22kg, Rivers State Nigeria, RDT positive, Hb 7.2, vomiting, cannot take oral medications.
Result: RED triage, IV Artesunate 2.4mg/kg, chloroquine resistance alert, blood glucose monitoring.

**Bacterial meningitis:** Adult with Kernig sign positive, petechial rash, haemodynamic compromise.
Result: RED triage, IV Ceftriaxone 2g STAT with Dexamethasone, DIC risk flagged.

**Pre-eclampsia:** 26-week pregnant woman, BP 158/104, neurological symptoms.
Result: RED triage, MgSO4 loading dose, calcium gluconate antidote alert, HELLP differential.

---

## Technology Stack

- **Backend:** Python 3.12, FastAPI, Uvicorn
- **AI inference:** Ollama, gemma3:4b
- **Knowledge retrieval:** ChromaDB (vector store RAG)
- **PDF generation:** ReportLab
- **Frontend:** Vanilla HTML, CSS, JavaScript (Progressive Web App)
- **Voice intake:** Web Speech API (local, no cloud)
- **Fine-tuning:** Unsloth LoRA on Gemma — Kaggle notebook included

---

## Fine-Tuning Pipeline

The repository includes a complete Kaggle fine-tuning notebook:

```
training/mediEdge_finetune_kaggle.ipynb
```

- Base model: Gemma 4B instruction-tuned
- Method: Unsloth LoRA (r=8, alpha=16)
- Training data: MedQA-USMLE + 5 gold-standard clinical cases
- Platform: Kaggle free tier T4 GPU
- Output: GGUF Q4_K_M for Ollama deployment

---

## Competition

Built for the **Gemma 4 Good Hackathon 2026**

- **Impact Track:** Health & Sciences
- **Special Technology Track:** Unsloth
- **Model:** gemma3:4b via Ollama (fully offline inference)

---

## Deployment Context

MediEdge was designed specifically for:

- Rural primary health centres across Rivers State and Niger Delta, Nigeria
- WHO-supported field hospitals in post-disaster zones
- Community health extension worker (CHEW) programmes across West Africa
- Refugee health camps with intermittent connectivity

The application installs on any Android or iOS phone from the browser — no app store required. No subscription. No server costs. No privacy risk — patient data never leaves the device.

---

## Repository Structure

```
mediedge/
├── backend/
│   ├── main.py              # FastAPI server — core clinical AI engine
│   └── requirements.txt     # Python dependencies
├── pwa/
│   └── public/
│       ├── index.html       # Complete PWA — all screens and logic
│       ├── manifest.json    # PWA manifest for mobile installation
│       └── sw.js            # Service worker for offline support
├── training/
│   └── mediEdge_finetune_kaggle.ipynb  # Unsloth LoRA fine-tuning notebook
└── README.md
```

---

## Author

**Anthony Mbadiwe Ikeme**

---

## License

Apache 2.0 — open for use in humanitarian and healthcare settings globally.
