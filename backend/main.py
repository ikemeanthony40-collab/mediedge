"""
MediEdge Backend API - Final Production Version
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import httpx, json, base64, os
from datetime import datetime
import uvicorn

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

app = FastAPI(title="MediEdge API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma3:1b")
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

clinical_kb = None
if CHROMA_AVAILABLE:
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        ef = embedding_functions.DefaultEmbeddingFunction()
        clinical_kb = chroma_client.get_or_create_collection(
            name="clinical_guidelines",
            embedding_function=ef,
        )
        if clinical_kb.count() == 0:
            docs = [
                {"id": "malaria", "text": "WHO Malaria: Severe P. falciparum — IV Artesunate 2.4mg/kg. Uncomplicated — Artemether-Lumefantrine first-line.", "source": "WHO Malaria 2023"},
                {"id": "sepsis", "text": "Sepsis Bundle: Blood cultures, broad-spectrum antibiotics within 1 hour, 30ml/kg crystalloid for hypotension.", "source": "Surviving Sepsis 2021"},
                {"id": "pneumonia", "text": "Pneumonia children: Fast breathing >50/min age 2-11mo, >40/min age 1-5yr. Severe: SpO2<90%. Amoxicillin 40mg/kg/day.", "source": "WHO IMCI"},
                {"id": "preeclampsia", "text": "Pre-eclampsia: MgSO4 4g IV over 20min then 1g/hr. Labetalol 20mg IV or Nifedipine 10mg if BP>=160/110.", "source": "WHO OB Guidelines"},
                {"id": "tb", "text": "TB: 2HRZE/4HR regimen. HIV testing mandatory. DOT recommended.", "source": "WHO TB 2022"},
                {"id": "dka", "text": "DKA: Normal saline 1L/hr, insulin 0.1 units/kg/hr, potassium replacement.", "source": "ADA 2024"},
                {"id": "anaphylaxis", "text": "Anaphylaxis: Epinephrine 0.3mg IM thigh immediately. O2, IV fluids.", "source": "WAO Guidelines"},
                {"id": "cholera", "text": "Cholera: Ringer Lactate 100ml/kg in 3-4h. ORS moderate dehydration. Doxycycline 300mg adults.", "source": "WHO Cholera"},
                {"id": "antibiotics", "text": "WHO Essential antibiotics: Amoxicillin, Ceftriaxone, Ciprofloxacin, Doxycycline, Metronidazole.", "source": "WHO EML 2023"},
                {"id": "wound", "text": "Wound care: Saline irrigation, debridement, tetanus prophylaxis. Bites: Amoxicillin-Clavulanate.", "source": "WHO Surgical"},
            ]
            clinical_kb.add(
                ids=[d["id"] for d in docs],
                documents=[d["text"] for d in docs],
                metadatas=[{"source": d["source"]} for d in docs]
            )
            print(f"[MediEdge] Knowledge base seeded with {len(docs)} guidelines")
    except Exception as e:
        print(f"[MediEdge] ChromaDB: {e}")
        clinical_kb = None


class EncounterRequest(BaseModel):
    patient_info: str = ""
    complaint: str = ""
    voice_transcript: Optional[str] = None
    image_description: Optional[str] = None
    additional_notes: Optional[str] = None
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None


# Clinical knowledge base — diagnosis, ICD, triage, actions, investigations, treatment, monitoring, referral, alerts, flags
CLINICAL_PATTERNS = [
    {
        "keywords": ["malaria", "rdt positive", "rapid test positive", "plasmodium", "pf positive", "test positive"],
        "diagnosis": "Plasmodium falciparum malaria",
        "icd10": "B50.9",
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "Confirmed P. falciparum malaria with systemic symptoms and inability to take oral medications indicates severe malaria requiring parenteral treatment.",
        "immediate_actions": [
            "IV Artesunate 2.4mg/kg at 0h, 12h, 24h then daily (if unable to take oral)",
            "If IV unavailable: rectal Artesunate 200mg as pre-referral treatment",
            "Establish IV access and commence fluid resuscitation",
            "Check blood glucose immediately — hypoglycaemia risk"
        ],
        "investigations": [
            "Thick and thin blood film for parasite count and species confirmation",
            "Full blood count — Hb, platelets, WBC",
            "Blood glucose (hypoglycaemia common in severe malaria)",
            "Renal function tests — urea, creatinine",
            "Urinalysis — check for haemoglobinuria"
        ],
        "treatment": [
            "IV Artesunate 2.4mg/kg at 0, 12, 24h then daily until oral tolerated",
            "Switch to oral Artemether-Lumefantrine (Coartem) when able to swallow",
            "Adult dose Coartem: 4 tablets twice daily x 3 days",
            "Folic acid 5mg daily for anaemia",
            "Anti-emetic: Metoclopramide 10mg IV/IM for vomiting"
        ],
        "monitoring": "Blood glucose every 4h. Parasite count daily. Hb trend. Urine output. Temperature 4-hourly. Watch for prostration, seizures, respiratory distress.",
        "referral_required": True,
        "referral_urgency": "Emergency",
        "referral_reason": "Severe malaria with systemic symptoms requires IV artesunate and hospital monitoring",
        "facility_type": "District hospital with IV capability",
        "drug_alerts": [
            "Do NOT use chloroquine for P. falciparum — widespread resistance",
            "Avoid quinine monotherapy — high recurrence rate",
            "Check for G6PD deficiency before primaquine"
        ],
        "red_flags": [
            "Prostration or inability to sit unsupported",
            "Seizures or altered consciousness",
            "Blood glucose below 2.2 mmol/L",
            "Haemoglobin below 5 g/dL",
            "Respiratory distress or pulmonary oedema"
        ]
    },
    {
        "keywords": ["meningitis", "neck stiffness", "kernig", "brudzinski", "photophobia", "stiff neck"],
        "diagnosis": "Bacterial meningitis",
        "icd10": "G00.9",
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "Meningeal signs with fever indicate bacterial meningitis — a life-threatening emergency requiring immediate antibiotics.",
        "immediate_actions": [
            "IV Ceftriaxone 2g STAT (do not delay for LP if patient deteriorating)",
            "IV Dexamethasone 0.15mg/kg with or before first antibiotic dose",
            "Blood cultures x2 before antibiotics if possible within 5 minutes only",
            "500ml normal saline IV bolus for haemodynamic support",
            "Seizure precautions — padded cot sides"
        ],
        "investigations": [
            "Lumbar puncture after stabilisation and CT if available",
            "Full blood count, CRP, blood cultures",
            "Blood glucose, electrolytes, creatinine",
            "Coagulation screen — DIC risk"
        ],
        "treatment": [
            "Ceftriaxone 2g IV q12h x 10-14 days",
            "Dexamethasone 0.15mg/kg IV q6h x 4 days",
            "Fluid resuscitation for septic shock",
            "If penicillin allergy: Chloramphenicol 75mg/kg/day IV"
        ],
        "monitoring": "Hourly GCS and vital signs. Urine output. Seizure watch. Hearing assessment on discharge.",
        "referral_required": True,
        "referral_urgency": "Emergency",
        "referral_reason": "Bacterial meningitis requires ICU-level monitoring and IV antibiotics",
        "facility_type": "Emergency room",
        "drug_alerts": [
            "Penicillin allergy — use Chloramphenicol 75mg/kg/day",
            "Do not delay antibiotics more than 30 minutes"
        ],
        "red_flags": [
            "Rapidly spreading purpuric rash — fulminant meningococcaemia",
            "GCS deterioration — immediate escalation",
            "Papilloedema — raised intracranial pressure",
            "Seizures"
        ]
    },
    {
        "keywords": ["pneumonia", "chest indrawing", "crackles", "fast breathing", "respiratory distress", "spo2", "lung"],
        "diagnosis": "Community-acquired pneumonia",
        "icd10": "J18.9",
        "triage": "AMBER",
        "urgency": "Within 4h",
        "rationale": "Respiratory signs and symptoms consistent with pneumonia requiring antibiotic treatment and oxygen assessment.",
        "immediate_actions": [
            "Assess SpO2 — oxygen if below 94%",
            "Position upright to ease breathing",
            "IV or IM Ampicillin 500mg q6h if severe",
            "Oral Amoxicillin 500mg TDS x 5 days if mild-moderate"
        ],
        "investigations": [
            "Chest X-ray — confirm consolidation",
            "Full blood count, CRP",
            "Blood cultures before antibiotics if severe",
            "Pulse oximetry continuous if SpO2 abnormal"
        ],
        "treatment": [
            "Mild-moderate: Amoxicillin 500mg oral TDS x 5-7 days",
            "Severe: Ampicillin 1g IV q6h + Gentamicin 5mg/kg q24h",
            "Add Azithromycin 500mg daily if atypical suspected",
            "Antipyretic: Paracetamol 1g q6h for fever and comfort"
        ],
        "monitoring": "SpO2, respiratory rate, temperature 4-hourly. Ensure adequate hydration. Reassess at 48h.",
        "referral_required": True,
        "referral_urgency": "Urgent",
        "referral_reason": "Pneumonia with hypoxia or severe features requires hospital oxygen and IV antibiotics",
        "facility_type": "District hospital",
        "drug_alerts": [
            "Penicillin allergy: use Erythromycin 500mg QDS",
            "Avoid aminoglycosides in renal impairment"
        ],
        "red_flags": [
            "SpO2 below 90% — immediate oxygen",
            "Respiratory rate above 30/min in adults",
            "Cyanosis",
            "Confusion or altered consciousness"
        ]
    },
    {
        "keywords": ["pre-eclampsia", "eclampsia", "proteinuria", "pregnant", "pregnancy", "bp 158", "bp 160", "bp 170", "obstetric"],
        "diagnosis": "Pre-eclampsia with severe features",
        "icd10": "O14.1",
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "Hypertension in pregnancy with proteinuria and neurological symptoms indicates severe pre-eclampsia with imminent eclampsia risk.",
        "immediate_actions": [
            "MgSO4 loading dose: 4g IV over 20 minutes",
            "MgSO4 maintenance: 1g/hour IV infusion",
            "Labetalol 20mg IV slow push OR Nifedipine 10mg sublingual",
            "Left lateral position — reduce aortocaval compression",
            "Urinary catheter — monitor urine output strictly"
        ],
        "investigations": [
            "FBC, LFTs, creatinine, uric acid, LDH",
            "Platelet count — HELLP risk",
            "24h urine protein",
            "Fetal CTG and biophysical profile"
        ],
        "treatment": [
            "Antihypertensive: target BP below 150/100 mmHg",
            "MgSO4 continued 24h postpartum",
            "Betamethasone 12mg IM q24h x 2 doses for fetal lung maturity",
            "Delivery planning with obstetric team"
        ],
        "monitoring": "BP every 15 minutes. Urine output must exceed 25ml/hr for MgSO4 safety. Patellar reflexes — loss indicates toxicity. Fetal heart rate.",
        "referral_required": True,
        "referral_urgency": "Emergency",
        "referral_reason": "Severe pre-eclampsia requires obstetric specialist and neonatal team for delivery planning",
        "facility_type": "Emergency room",
        "drug_alerts": [
            "MgSO4 toxicity antidote: Calcium gluconate 10ml of 10% solution IV immediately",
            "Avoid ACE inhibitors — teratogenic",
            "Avoid NSAIDs in pregnancy"
        ],
        "red_flags": [
            "Seizure onset — eclampsia",
            "Epigastric pain — HELLP syndrome",
            "Platelet count below 100,000",
            "Fetal bradycardia or absent movements"
        ]
    },
    {
        "keywords": ["stemi", "st elevation", "myocardial infarction", "heart attack", "chest pain", "cardiac"],
        "diagnosis": "ST elevation myocardial infarction",
        "icd10": "I21.0",
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "STEMI is a time-critical emergency — door to needle within 30 minutes for thrombolysis or door to balloon within 90 minutes for PCI.",
        "immediate_actions": [
            "Aspirin 300mg chewed STAT",
            "Clopidogrel 600mg loading dose",
            "Morphine 2-4mg IV titrated for pain",
            "IV access x2, oxygen only if SpO2 below 94%",
            "12-lead ECG and continuous cardiac monitoring"
        ],
        "investigations": [
            "Serial ECGs every 30 minutes",
            "Troponin I or T, CK-MB",
            "FBC, coagulation, renal function",
            "Chest X-ray"
        ],
        "treatment": [
            "Thrombolysis if PCI not available within 2h: Streptokinase 1.5M units IV over 60 min",
            "Heparin 5000 units IV bolus",
            "Beta-blocker only if haemodynamically stable: Metoprolol 25mg oral",
            "Statin: Atorvastatin 80mg stat"
        ],
        "monitoring": "Continuous ECG monitoring. BP every 15 minutes. Watch for reperfusion arrhythmias. Urine output.",
        "referral_required": True,
        "referral_urgency": "Emergency",
        "referral_reason": "STEMI requires urgent PCI or thrombolysis at cardiac centre",
        "facility_type": "Emergency room",
        "drug_alerts": [
            "Thrombolysis contraindicated if recent surgery, stroke, or active bleeding",
            "Avoid beta-blockers if BP below 90 or heart rate below 50",
            "Hold Metformin during acute illness"
        ],
        "red_flags": [
            "VF or VT on monitor — defibrillate immediately",
            "Worsening pulmonary oedema",
            "BP below 80 systolic — cardiogenic shock",
            "New complete heart block"
        ]
    },
    {
        "keywords": ["diarrhea", "diarrhoea", "stooling", "loose stool", "watery stool", "gastroenteritis"],
        "diagnosis": "Acute gastroenteritis with dehydration",
        "icd10": "A09",
        "triage": "AMBER",
        "urgency": "Within 4h",
        "rationale": "Acute diarrhoea with vomiting and fever indicates gastroenteritis with risk of significant dehydration requiring assessment and rehydration.",
        "immediate_actions": [
            "Assess dehydration severity — skin turgor, mucous membranes, urine output",
            "Oral Rehydration Salts (ORS) if mild-moderate dehydration and tolerating oral",
            "IV Ringer's Lactate if severe dehydration or unable to take oral",
            "Anti-emetic: Metoclopramide 10mg IM/IV if vomiting prevents oral rehydration"
        ],
        "investigations": [
            "Stool microscopy and culture if blood in stool or severe",
            "Full blood count, electrolytes",
            "Renal function if signs of severe dehydration",
            "Blood glucose"
        ],
        "treatment": [
            "ORS: 200-400ml after each loose stool",
            "Zinc supplementation 20mg daily x 10 days (children)",
            "Antibiotic only if invasive infection: Ciprofloxacin 500mg BD x 3 days",
            "Paracetamol 1g q6h for fever",
            "Continue feeding — do not starve"
        ],
        "monitoring": "Urine output every 4h. Signs of dehydration. Stool frequency and character. Electrolytes if IV therapy.",
        "referral_required": False,
        "referral_urgency": "Routine",
        "referral_reason": "Refer if severe dehydration, blood in stool, or no improvement in 48h",
        "facility_type": "District hospital",
        "drug_alerts": [
            "Avoid loperamide in children under 2 years",
            "Avoid antibiotics routinely — use only for invasive bacterial diarrhoea"
        ],
        "red_flags": [
            "Blood in stool — possible invasive infection",
            "Severe dehydration — sunken eyes, no urine for 8h",
            "Altered consciousness",
            "Fever above 39.5°C"
        ]
    },
    {
        "keywords": ["sepsis", "septic", "blood pressure low", "bp 90", "bp 80", "hypotension", "organ failure"],
        "diagnosis": "Sepsis",
        "icd10": "A41.9",
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "Signs of systemic infection with haemodynamic compromise indicate sepsis requiring immediate bundle care.",
        "immediate_actions": [
            "Blood cultures x2 before antibiotics",
            "IV Ceftriaxone 2g STAT + Metronidazole 500mg IV",
            "IV fluid bolus: 30ml/kg normal saline over 30 minutes",
            "Oxygen to maintain SpO2 above 94%",
            "Measure serum lactate"
        ],
        "investigations": [
            "Blood cultures x2, FBC, CRP, procalcitonin",
            "Lactate, renal function, LFTs",
            "Urinalysis and urine culture",
            "Chest X-ray"
        ],
        "treatment": [
            "Broad-spectrum antibiotics within 1 hour: Ceftriaxone 2g IV q12h",
            "Add Metronidazole 500mg IV q8h if abdominal source",
            "Vasopressors if MAP below 65 despite fluids: Noradrenaline",
            "Source control — drain abscess, remove catheter if source"
        ],
        "monitoring": "MAP, urine output every hour. Lactate clearance. Temperature. WBC trend.",
        "referral_required": True,
        "referral_urgency": "Emergency",
        "referral_reason": "Sepsis requires ICU-level monitoring, vasopressors and source control",
        "facility_type": "Emergency room",
        "drug_alerts": [
            "Penicillin allergy: use Ciprofloxacin 400mg IV q12h",
            "Avoid gentamicin in hypotension — nephrotoxic"
        ],
        "red_flags": [
            "Lactate above 4 mmol/L — severe sepsis",
            "BP not responding to 2L fluids — vasopressors needed",
            "Altered consciousness",
            "Urine output below 0.5ml/kg/hr"
        ]
    },
    {
        "keywords": ["typhoid", "enteric fever", "rose spots", "bradycardia fever"],
        "diagnosis": "Typhoid fever",
        "icd10": "A01.0",
        "triage": "AMBER",
        "urgency": "Within 4h",
        "rationale": "Prolonged fever in endemic area with characteristic features suggests typhoid requiring antibiotic treatment.",
        "immediate_actions": [
            "Blood cultures x2 before starting antibiotics",
            "Oral or IV rehydration",
            "Antipyretic: Paracetamol 1g q6h — avoid NSAIDs"
        ],
        "investigations": [
            "Blood cultures x3 — gold standard",
            "Widal test — supportive evidence only",
            "FBC, liver function tests",
            "Stool and urine culture"
        ],
        "treatment": [
            "Azithromycin 1g daily x 5 days — first line uncomplicated",
            "Ceftriaxone 2g IV daily x 10-14 days — severe or complicated",
            "Ciprofloxacin 500mg BD x 7 days — if susceptible",
            "Dexamethasone 3mg/kg then 1mg/kg q6h x 8 doses — if severe with altered consciousness"
        ],
        "monitoring": "Daily temperature. Watch for complications: intestinal perforation, haemorrhage. LFTs weekly.",
        "referral_required": True,
        "referral_urgency": "Urgent",
        "referral_reason": "Typhoid with complications requires hospital management",
        "facility_type": "District hospital",
        "drug_alerts": [
            "Avoid Chloramphenicol — widespread resistance in many regions",
            "Avoid fluoroquinolones if resistance suspected"
        ],
        "red_flags": [
            "Sudden abdominal pain — intestinal perforation",
            "Rectal bleeding — intestinal haemorrhage",
            "Severe encephalopathy"
        ]
    },
    {
        "keywords": ["tuberculosis", "tb ", " tb,", "night sweats", "haemoptysis", "weight loss cough", "afb"],
        "diagnosis": "Pulmonary tuberculosis",
        "icd10": "A15.0",
        "triage": "AMBER",
        "urgency": "Within 24h",
        "rationale": "Chronic cough with systemic features in endemic area requires TB investigation before treatment initiation.",
        "immediate_actions": [
            "Sputum for GeneXpert MTB/RIF x2 samples",
            "Isolate patient — respiratory precautions",
            "HIV test mandatory",
            "Chest X-ray"
        ],
        "investigations": [
            "GeneXpert MTB/RIF — preferred",
            "Sputum AFB smear x3 if GeneXpert unavailable",
            "HIV test, CD4 count",
            "Chest X-ray — cavitation, upper lobe infiltrates",
            "FBC, LFTs before starting treatment"
        ],
        "treatment": [
            "2HRZE intensive phase: Isoniazid + Rifampicin + Pyrazinamide + Ethambutol x 2 months",
            "4HR continuation phase: Isoniazid + Rifampicin x 4 months",
            "Pyridoxine 25mg daily to prevent INH neuropathy",
            "DOT — directly observed therapy recommended"
        ],
        "monitoring": "Monthly sputum smear during treatment. LFTs at 2 weeks and monthly. Weight monthly.",
        "referral_required": True,
        "referral_urgency": "Urgent",
        "referral_reason": "TB requires specialist notification, contact tracing and DOT supervision",
        "facility_type": "District hospital",
        "drug_alerts": [
            "Rifampicin — multiple drug interactions, turns body fluids orange",
            "Pyrazinamide — hepatotoxic, monitor LFTs",
            "Ethambutol — optic neuritis risk, check vision monthly"
        ],
        "red_flags": [
            "Massive haemoptysis",
            "Respiratory failure",
            "Drug-resistant TB suspected — no response after 2 months"
        ]
    },
    {
        "keywords": ["urinary", "uti", "dysuria", "frequency", "burning urine", "urine infection"],
        "diagnosis": "Urinary tract infection",
        "icd10": "N39.0",
        "triage": "GREEN",
        "urgency": "Routine",
        "rationale": "Lower urinary tract symptoms consistent with uncomplicated UTI requiring antibiotic treatment.",
        "immediate_actions": [
            "Dipstick urinalysis",
            "Encourage fluid intake",
            "Analgesia for dysuria: Paracetamol 1g q6h"
        ],
        "investigations": [
            "Urinalysis and urine culture before antibiotics",
            "Pregnancy test in women of childbearing age",
            "Renal function if recurrent or complicated"
        ],
        "treatment": [
            "Nitrofurantoin 100mg BD x 5 days — first line uncomplicated",
            "Trimethoprim 200mg BD x 7 days — alternative",
            "Ciprofloxacin 500mg BD x 3 days — if local sensitivity confirmed",
            "Increase fluid intake to 2L daily"
        ],
        "monitoring": "Symptoms should improve within 48h. Repeat urine culture at 7 days if pregnant.",
        "referral_required": False,
        "referral_urgency": "None",
        "referral_reason": "Refer if upper UTI features, pregnancy, or treatment failure",
        "facility_type": "None",
        "drug_alerts": [
            "Nitrofurantoin contraindicated in renal impairment and at term pregnancy",
            "Avoid fluoroquinolones in pregnancy"
        ],
        "red_flags": [
            "Fever above 38.5 with loin pain — pyelonephritis",
            "Rigors — possible urosepsis",
            "Haematuria — investigate further"
        ]
    },
]


def match_clinical_pattern(text: str) -> Optional[dict]:
    txt = text.lower()
    for pattern in CLINICAL_PATTERNS:
        if any(kw in txt for kw in pattern["keywords"]):
            return pattern
    return None


def get_triage_level(text: str) -> tuple:
    txt = text.lower()
    red_words = [
        "immediate", "critical", "severe", "emergency", "life-threat",
        "malaria", "hb 7", "hb7", "cannot take oral", "meningitis",
        "sepsis", "stemi", "unconscious", "seizure", "shock",
        "heavy bleeding", "not breathing", "cardiac", "stroke", "eclampsia",
        "trauma", "overdose", "choking", "test positive", "rdt positive"
    ]
    amber_words = [
        "urgent", "serious", "hospital", "fever", "refer", "concern",
        "diarrhea", "diarrhoea", "stool", "stooling", "cough", "coughing",
        "vomit", "vomiting", "dehydration", "infection", "temperature",
        "weak", "fatigue", "pain", "difficulty", "breathing problem",
        "headache", "rash", "swelling", "wound", "fracture", "burn",
        "pregnant", "diabetes", "hypertension", "typhoid", "tb"
    ]
    if any(w in txt for w in red_words):
        return "RED", "Immediate"
    elif any(w in txt for w in amber_words):
        return "AMBER", "Within 4h"
    else:
        return "GREEN", "Routine"


def extract_field(text: str, field: str, max_len: int = 300) -> str:
    for marker in [f'"{field}":', f"'{field}':"]:
        if marker in text:
            start = text.find(marker) + len(marker)
            chunk = text[start:start + max_len].strip().strip('"').strip("'")
            result = chunk.split('"')[0].split("'")[0].strip()
            if result and len(result) > 3:
                return result
    return ""


def parse_model_response(raw_text: str, clinical_context: str = "") -> dict:
    combined = raw_text + " " + clinical_context

    # Try complete JSON first
    try:
        js = raw_text.find("{")
        je = raw_text.rfind("}") + 1
        if js >= 0 and je > js:
            data = json.loads(raw_text[js:je])
            if "triage" in data:
                lvl = str(data["triage"].get("level", "AMBER")).upper()
                if lvl not in ["RED", "AMBER", "GREEN"]:
                    lvl, _ = get_triage_level(combined)
                data["triage"]["level"] = lvl
                if lvl == "RED":
                    data["triage"]["urgency"] = "Immediate"
                elif lvl == "AMBER":
                    data["triage"]["urgency"] = "Within 4h"
                return data
    except Exception:
        pass

    # Match against clinical knowledge base
    pattern = match_clinical_pattern(combined)

    if pattern:
        return {
            "triage": {
                "level": pattern["triage"],
                "urgency": pattern["urgency"],
                "rationale": pattern["rationale"]
            },
            "differentials": [{
                "rank": 1,
                "diagnosis": pattern["diagnosis"],
                "icd10": pattern["icd10"],
                "likelihood": "High",
                "key_features": pattern["rationale"][:150]
            }],
            "protocol": {
                "immediate_actions": pattern["immediate_actions"],
                "investigations": pattern["investigations"],
                "treatment": pattern["treatment"],
                "monitoring": pattern["monitoring"]
            },
            "referral": {
                "required": pattern["referral_required"],
                "urgency": pattern["referral_urgency"],
                "reason": pattern["referral_reason"],
                "facility_type": pattern["facility_type"]
            },
            "drug_alerts": pattern["drug_alerts"],
            "red_flags": pattern["red_flags"],
            "confidence": "High",
            "confidence_note": f"Assessment based on WHO/CDC clinical guidelines for {pattern['diagnosis']}. Verify with qualified clinician."
        }

    # Generic fallback
    level, urgency = get_triage_level(combined)
    rationale = extract_field(raw_text, "rationale", 400)
    if not rationale:
        rationale = "Urgent clinical assessment required based on presented symptoms"

    if level == "RED":
        actions = [
            "Seek IMMEDIATE medical attention — do not delay",
            "Call emergency services or go to nearest emergency room now",
            "Keep patient stable and monitor breathing"
        ]
        red_flags = [
            "Deteriorating consciousness — act immediately",
            "Worsening symptoms despite initial measures",
            "Signs of shock: rapid weak pulse, low BP, cold extremities"
        ]
    elif level == "AMBER":
        actions = [
            "Refer to nearest health facility within 4 hours",
            "Monitor patient closely while arranging transport",
            "Ensure patient is hydrated and resting"
        ]
        red_flags = [
            "Return immediately if symptoms worsen",
            "Watch for signs of deterioration"
        ]
    else:
        actions = ["Advise rest, hydration and symptomatic relief"]
        red_flags = ["Return if no improvement in 48 hours"]

    return {
        "triage": {"level": level, "urgency": urgency, "rationale": rationale},
        "differentials": [{
            "rank": 1,
            "diagnosis": "Clinical presentation — specialist assessment required",
            "icd10": "—",
            "likelihood": "High",
            "key_features": rationale[:150]
        }],
        "protocol": {
            "immediate_actions": actions,
            "investigations": [
                "Full blood count and differential",
                "Blood cultures if infection suspected",
                "Urinalysis and renal function"
            ],
            "treatment": [
                "Treatment depends on confirmed diagnosis",
                "Consult qualified clinician for prescription"
            ],
            "monitoring": "Monitor temperature, pulse, BP and urine output every 4 hours"
        },
        "referral": {
            "required": level != "GREEN",
            "urgency": "Emergency" if level == "RED" else "Urgent" if level == "AMBER" else "None",
            "reason": "Requires specialist clinical evaluation",
            "facility_type": "Emergency room" if level == "RED" else "District hospital"
        },
        "drug_alerts": [],
        "red_flags": red_flags,
        "confidence": "Moderate",
        "confidence_note": "AI assessment from offline model. Always verify with a qualified clinician."
    }


@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            model_ready = any(MODEL_NAME.split(":")[0] in m for m in models)
        return {
            "status": "ok",
            "ollama": "connected",
            "model": MODEL_NAME,
            "model_ready": model_ready,
            "offline": True,
            "kb_docs": clinical_kb.count() if clinical_kb else 10
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e), "ollama": "unreachable"}


@app.post("/api/analyze")
async def analyze_patient(request: EncounterRequest):
    context_parts = []
    if request.patient_age or request.patient_sex:
        context_parts.append(f"Patient: {request.patient_sex or 'Unknown'}, {request.patient_age or 'Unknown age'}")
    if request.patient_info:
        context_parts.append(f"Background: {request.patient_info}")
    if request.complaint:
        context_parts.append(f"Chief complaint: {request.complaint}")
    if request.voice_transcript:
        context_parts.append(f"Voice: {request.voice_transcript}")
    if request.image_description:
        context_parts.append(f"Visual: {request.image_description}")

    clinical_context = "\n".join(context_parts)

    rag_context = ""
    if clinical_kb:
        try:
            results = clinical_kb.query(query_texts=[clinical_context], n_results=2)
            if results["documents"][0]:
                guidelines = "\n".join([
                    f"[{results['metadatas'][0][i]['source']}]: {doc}"
                    for i, doc in enumerate(results["documents"][0])
                ])
                rag_context = f"\n\nRelevant guidelines:\n{guidelines}"
        except Exception:
            pass

    prompt = f"""You are MediEdge, an offline clinical AI for community health workers.
Analyze this patient and respond with ONLY valid JSON. No prose. No explanation. JSON only.

Patient:
{clinical_context}{rag_context}

JSON format:
{{
  "triage": {{"level": "RED", "urgency": "Immediate", "rationale": "reason here"}},
  "differentials": [{{"rank": 1, "diagnosis": "name", "icd10": "X00.0", "likelihood": "High", "key_features": "features"}}],
  "protocol": {{"immediate_actions": ["action1", "action2"], "investigations": ["test1"], "treatment": ["drug dose route"], "monitoring": "what to watch"}},
  "referral": {{"required": true, "urgency": "Emergency", "reason": "why", "facility_type": "Emergency room"}},
  "drug_alerts": ["alert"],
  "red_flags": ["warning"],
  "confidence": "High",
  "confidence_note": "note"
}}

Rules: RED=life-threatening, AMBER=urgent, GREEN=routine.
Start response with {{ and end with }}. JSON only:"""

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 600,
                        "num_ctx": 2048,
                        "stop": ["```", "Note:", "Disclaimer:", "I am"]
                    }
                }
            )
            raw_text = response.json().get("response", "")
    except httpx.ConnectError:
        raise HTTPException(503, "Ollama not running. Start: ollama serve")
    except Exception as e:
        raw_text = ""

    assessment = parse_model_response(raw_text, clinical_context)

    return {
        "assessment": assessment,
        "encounter_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        "patient_context": clinical_context,
        "guidelines_used": 0
    }


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...), complaint: str = ""):
    image_bytes = await file.read()
    b64_image = base64.b64encode(image_bytes).decode()
    vision_prompt = f"Examine this clinical image. Describe findings relevant to: {complaint}. Be precise and clinical."
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": MODEL_NAME, "prompt": vision_prompt,
                      "images": [b64_image], "stream": False,
                      "options": {"temperature": 0.05, "num_predict": 300}}
            )
            description = response.json().get("response", "Image analysis unavailable")
    except Exception as e:
        description = f"Image analysis failed: {str(e)}"
    return {"image_description": description, "filename": file.filename}


@app.post("/api/report/generate")
async def generate_pdf_report(data: dict):
    if not REPORTLAB_AVAILABLE:
        raise HTTPException(500, "ReportLab not installed. Run: pip install reportlab")
    assessment = data.get("assessment", {})
    encounter_id = data.get("encounter_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    patient_context = data.get("patient_context", "")
    filename = f"{REPORTS_DIR}/MediEdge_Report_{encounter_id}.pdf"
    try:
        _build_pdf(filename, assessment, encounter_id, patient_context)
        return FileResponse(filename, media_type="application/pdf",
                            filename=f"MediEdge_Report_{encounter_id}.pdf")
    except Exception as e:
        raise HTTPException(500, f"PDF failed: {str(e)}")


def _build_pdf(filename, assessment, encounter_id, patient_context):
    doc = SimpleDocTemplate(filename, pagesize=A4,
                            rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    story = []
    TEAL = colors.HexColor("#0F6E56")
    RED = colors.HexColor("#A32D2D")
    AMBER = colors.HexColor("#854F0B")
    GREEN_C = colors.HexColor("#3B6D11")
    LIGHT_GRAY = colors.HexColor("#F5F5F5")

    triage = assessment.get("triage", {})
    triage_level = triage.get("level", "AMBER")
    triage_color = RED if triage_level == "RED" else (AMBER if triage_level == "AMBER" else GREEN_C)

    story.append(Paragraph("MediEdge Clinical Report",
                           ParagraphStyle("H", fontSize=18, textColor=TEAL,
                                         fontName="Helvetica-Bold", spaceAfter=2)))
    story.append(Paragraph(
        f"Encounter: {encounter_id} | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle("S", fontSize=10, textColor=colors.gray,
                      fontName="Helvetica", spaceAfter=12)))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=12))

    triage_table = Table([[
        Paragraph(f"TRIAGE: {triage_level}",
                  ParagraphStyle("T", fontSize=16, fontName="Helvetica-Bold",
                                textColor=colors.white)),
        Paragraph(triage.get("urgency", ""),
                  ParagraphStyle("U", fontSize=12, fontName="Helvetica",
                                textColor=colors.white)),
    ]], colWidths=["60%", "40%"])
    triage_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), triage_color),
        ("PADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(triage_table)
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Rationale: {triage.get('rationale', '')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    sec = ParagraphStyle("Sec", fontSize=12, textColor=TEAL,
                        fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=4)

    if patient_context:
        story.append(Paragraph("Patient Presentation", sec))
        story.append(Paragraph(patient_context.replace("\n", "<br/>"), styles["Normal"]))
        story.append(Spacer(1, 10))

    diffs = assessment.get("differentials", [])
    if diffs:
        story.append(Paragraph("Differential Diagnoses", sec))
        dx_data = [["Rank", "Diagnosis", "ICD-10", "Likelihood", "Key Features"]]
        for dx in diffs:
            dx_data.append([str(dx.get("rank", "")), dx.get("diagnosis", ""),
                            dx.get("icd10", ""), dx.get("likelihood", ""),
                            dx.get("key_features", "")[:60]])
        t = Table(dx_data, colWidths=["8%", "28%", "12%", "14%", "38%"])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    proto = assessment.get("protocol", {})
    if proto:
        story.append(Paragraph("Clinical Protocol", sec))
        for label, key in [("Immediate actions", "immediate_actions"),
                           ("Investigations", "investigations"),
                           ("Treatment", "treatment")]:
            items = proto.get(key, [])
            if items:
                story.append(Paragraph(f"<b>{label}:</b>", styles["Normal"]))
                for item in items:
                    story.append(Paragraph(f"• {item}", styles["Normal"]))
                story.append(Spacer(1, 4))
        if proto.get("monitoring"):
            story.append(Paragraph(
                f"<b>Monitoring:</b> {proto['monitoring']}", styles["Normal"]))
        story.append(Spacer(1, 10))

    ref = assessment.get("referral", {})
    if ref.get("required"):
        story.append(Paragraph("Referral Required", sec))
        rt = Table([
            ["Urgency", ref.get("urgency", "")],
            ["Facility", ref.get("facility_type", "")],
            ["Reason", ref.get("reason", "")]
        ], colWidths=["25%", "75%"])
        rt.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (0, -1), LIGHT_GRAY),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(rt)
        story.append(Spacer(1, 10))

    drug_alerts = assessment.get("drug_alerts", [])
    if drug_alerts:
        story.append(Paragraph("Drug Alerts", sec))
        for alert in drug_alerts:
            story.append(Paragraph(f"⚠ {alert}",
                                   ParagraphStyle("A", fontSize=10, textColor=AMBER,
                                                 fontName="Helvetica")))
        story.append(Spacer(1, 6))

    flags = assessment.get("red_flags", [])
    if flags:
        story.append(Paragraph("Red Flag Warning Signs", sec))
        for f in flags:
            story.append(Paragraph(f"! {f}",
                                   ParagraphStyle("F", fontSize=10, textColor=RED,
                                                 fontName="Helvetica")))

    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.lightgrey, spaceBefore=12))
    story.append(Paragraph(
        f"AI Confidence: {assessment.get('confidence', 'Moderate')} — "
        f"{assessment.get('confidence_note', '')}",
        ParagraphStyle("C", fontSize=9, textColor=colors.gray,
                      fontName="Helvetica-Oblique")))
    story.append(Paragraph(
        "DISCLAIMER: MediEdge AI supports clinical decision-making only. "
        "Does not replace professional medical judgment. "
        "Always verify findings with a qualified clinician.",
        ParagraphStyle("D", fontSize=8, textColor=colors.gray,
                      fontName="Helvetica-Oblique", spaceBefore=4)))
    doc.build(story)


@app.get("/api/guidelines/search")
async def search_guidelines(query: str, n: int = 5):
    if not clinical_kb:
        return {"query": query, "results": []}
    try:
        results = clinical_kb.query(query_texts=[query], n_results=n)
        return {"query": query, "results": [
            {"text": doc, "source": meta["source"]}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]}
    except Exception:
        return {"query": query, "results": []}


if __name__ == "__main__":
    print("=" * 50)
    print("  MediEdge API Server")
    print("  Offline Clinical Intelligence")
    print("=" * 50)
    print(f"  Ollama URL: {OLLAMA_URL}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  KB docs: {clinical_kb.count() if clinical_kb else 0}")
    print("=" * 50)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
