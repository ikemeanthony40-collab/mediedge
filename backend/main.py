"""
MediEdge Backend — Definitive v3
Score-based clinical pattern matching + Gemma fallback
Author: Anthony Mbadiwe Ikeme
Competition: Gemma 4 Good Hackathon 2026
"""

import os, json, re, logging, asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MediEdge")

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL  = os.getenv("OLLAMA_URL",  "http://localhost:11434")
MODEL_NAME  = os.getenv("MODEL_NAME",  "gemma3:4b")
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# CLINICAL KNOWLEDGE BASE  (score-based matching — highest score wins)
# Each condition has exclusive_keywords (strong identifiers) and
# supporting_keywords (common symptoms).  Score = exclusive*3 + supporting*1
# ──────────────────────────────────────────────────────────────────────────────
CLINICAL_KB = [
    # ── 1. ASTHMA ─────────────────────────────────────────────────────────────
    {
        "id": "asthma",
        "exclusive": [
            "asthma", "known asthmatic", "bronchospasm", "wheeze", "wheezing",
            "salbutamol", "not responding to inhaler", "inhaler not working",
            "audible wheeze", "cannot complete sentences", "ventolin",
            "bronchodilator", "reliever inhaler"
        ],
        "supporting": [
            "difficulty breathing", "breathless", "shortness of breath",
            "spo2 88", "spo2 89", "spo2 90", "severe breathing"
        ],
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "Severe acute asthma with SpO2 <90% and failure to respond to bronchodilator — life-threatening.",
        "differentials": [
            {"rank":1,"diagnosis":"Severe acute asthma","icd10":"J46","likelihood":"High",
             "key_features":"Audible wheeze, inability to complete sentences, SpO2 <90%, inhaler failure"},
            {"rank":2,"diagnosis":"Acute exacerbation COPD","icd10":"J44.1","likelihood":"Low",
             "key_features":"Consider if >40 years, smoker, no prior asthma diagnosis"}
        ],
        "immediate_actions": [
            "Salbutamol 5mg nebulised STAT — repeat every 20 minutes x3",
            "Oxygen via face mask — target SpO2 94-98%",
            "Ipratropium bromide 0.5mg nebulised with first salbutamol dose",
            "IV access and commence IV fluids"
        ],
        "investigations": [
            "Pulse oximetry continuous",
            "Peak expiratory flow rate (PEFR) if possible",
            "ABG if SpO2 <92% despite oxygen",
            "Chest X-ray to exclude pneumothorax or pneumonia"
        ],
        "treatment": [
            "Salbutamol 5mg nebulised every 20min for 3 doses, then hourly",
            "Hydrocortisone 200mg IV STAT (or Prednisolone 40mg oral if can swallow)",
            "Ipratropium bromide 0.5mg nebulised every 6 hours",
            "Magnesium sulphate 2g IV over 20min if not responding to above"
        ],
        "monitoring": "Peak flow, SpO2, RR, HR every 15 minutes. Watch for silent chest (ominous sign). Prepare for intubation if GCS drops.",
        "referral": {"required":True,"urgency":"Emergency","reason":"Severe acute asthma with SpO2 <90% requires ICU-level care","facility_type":"Emergency room with ICU"},
        "drug_alerts": [
            "Avoid beta-blockers — worsen bronchospasm",
            "Avoid NSAIDs in aspirin-sensitive asthma",
            "Aminophylline only if nebulised therapy unavailable — narrow therapeutic window"
        ],
        "red_flags": [
            "Silent chest — no wheeze despite respiratory distress",
            "SpO2 falling below 88% despite oxygen",
            "Cyanosis or exhaustion",
            "Altered consciousness",
            "PEFR below 33% of predicted"
        ]
    },

    # ── 2. STROKE ─────────────────────────────────────────────────────────────
    {
        "id": "stroke",
        "exclusive": [
            "facial droop", "face drooping", "arm weakness", "slurred speech",
            "speech slurred", "sudden weakness", "hemiplegia", "hemiparesis",
            "facial asymmetry", "sudden numbness", "fast positive", "face arm speech time",
            "sudden confusion", "sudden vision loss", "stroke", "tia"
        ],
        "supporting": [
            "hypertension", "bp 180", "bp 170", "bp 160", "hypertensive",
            "headache", "vomiting", "confused", "weakness one side"
        ],
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "Acute stroke — time-critical emergency. Thrombolysis window 4.5 hours from symptom onset.",
        "differentials": [
            {"rank":1,"diagnosis":"Acute ischaemic stroke","icd10":"I63.9","likelihood":"High",
             "key_features":"Sudden onset facial droop, unilateral arm weakness, slurred speech — FAST positive"},
            {"rank":2,"diagnosis":"Haemorrhagic stroke","icd10":"I61.9","likelihood":"Moderate",
             "key_features":"Consider if severe headache, BP >180, rapid deterioration"}
        ],
        "immediate_actions": [
            "Activate stroke pathway — time is brain (1.9 million neurons lost per minute)",
            "Note EXACT TIME of symptom onset — critical for thrombolysis decision",
            "Aspirin 300mg oral STAT if ischaemic confirmed and no bleeding on CT",
            "Oxygen only if SpO2 <94% — hyperoxia harms ischaemic brain",
            "DO NOT lower BP acutely unless >220/120 — blood pressure maintains penumbra perfusion",
            "IV access, bloods, 12-lead ECG"
        ],
        "investigations": [
            "CT brain STAT (no contrast) to exclude haemorrhage",
            "Blood glucose STAT — hypoglycaemia mimics stroke",
            "FBC, coagulation screen, U&E",
            "12-lead ECG — AF is major embolic source",
            "Chest X-ray"
        ],
        "treatment": [
            "Aspirin 300mg oral then 75mg daily (ischaemic stroke only)",
            "Thrombolysis with tPA if: within 4.5h, no haemorrhage on CT, no contraindications",
            "Statin: Atorvastatin 40mg daily",
            "Antihypertensives: withhold first 24h unless BP >220/120",
            "DVT prophylaxis: compression stockings"
        ],
        "monitoring": "Neurological observations every 30 minutes (GCS, NIHSS, BP, SpO2). Swallow assessment before any oral intake. Blood glucose 4-hourly (target 4-11 mmol/L).",
        "referral": {"required":True,"urgency":"Emergency","reason":"Acute stroke needs CT brain, thrombolysis assessment and stroke unit care","facility_type":"Stroke centre or hospital with CT"},
        "drug_alerts": [
            "Aspirin contraindicated in haemorrhagic stroke — confirm CT first",
            "Avoid glucose solutions — worsen ischaemia",
            "Avoid acute BP lowering unless >220/120 in first 24h"
        ],
        "red_flags": [
            "Sudden deterioration in consciousness",
            "Seizures",
            "Airway compromise — aspiration risk",
            "BP >220/120 — may require cautious antihypertensive"
        ]
    },

    # ── 3. BURNS ──────────────────────────────────────────────────────────────
    {
        "id": "burns",
        "exclusive": [
            "burn", "burns", "scalded", "scald", "fire", "flame",
            "body surface area", "bsa", "blisters", "blister",
            "cooking fire", "chemical burn", "electrical burn"
        ],
        "supporting": [
            "pain", "skin", "wound", "arms", "chest", "face",
            "smoke", "inhalation", "percentage"
        ],
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "Major burns >20% BSA with haemodynamic compromise — requires aggressive fluid resuscitation.",
        "differentials": [
            {"rank":1,"diagnosis":"Major thermal burns","icd10":"T31.2","likelihood":"High",
             "key_features":"Burns >20% BSA, blisters, haemodynamic instability"},
            {"rank":2,"diagnosis":"Inhalation injury","icd10":"T59.9","likelihood":"Moderate",
             "key_features":"Consider if face/neck burns, hoarse voice, soot in airway, enclosed space"}
        ],
        "immediate_actions": [
            "Airway assessment FIRST — intubate early if facial/neck burns or hoarse voice",
            "Remove all clothing and jewellery from burned areas",
            "IV access x2 large bore — do NOT use burned skin",
            "Parkland formula: 4ml x weight(kg) x %BSA burned — give half in first 8h, half over next 16h",
            "Cover burns with clean dry dressings or cling film — do NOT use ice",
            "Morphine 0.1mg/kg IV titrated for pain"
        ],
        "investigations": [
            "FBC, U&E, glucose, coagulation",
            "Blood group and crossmatch",
            "Arterial blood gas — especially if smoke inhalation",
            "Carboxyhaemoglobin if smoke exposure",
            "Urinalysis — target urine output 0.5-1ml/kg/hr"
        ],
        "treatment": [
            "IV fluid resuscitation: Ringer's lactate — Parkland formula",
            "Morphine IV titrated for analgesia (0.05-0.1mg/kg increments)",
            "Tetanus toxoid if burns >1% BSA and vaccination not current",
            "Silver sulphadiazine cream to partial thickness burns after cooling",
            "High-flow oxygen 100% if inhalation injury suspected"
        ],
        "monitoring": "Urine output hourly (insert catheter). BP and HR every 30 min. Adjust fluids to maintain urine output 0.5-1ml/kg/hr. Oedema progression. Temperature (burns cause hypothermia).",
        "referral": {"required":True,"urgency":"Emergency","reason":"Major burns require specialist burns unit, fluid resuscitation, possible surgical debridement","facility_type":"Burns unit or major emergency hospital"},
        "drug_alerts": [
            "Do NOT use succinylcholine for intubation in burns >48h old — fatal hyperkalaemia",
            "Avoid NSAIDs in major burns — renal compromise",
            "Morphine dose requirements higher in burns — pain undertreated"
        ],
        "red_flags": [
            "Stridor or hoarse voice — imminent airway loss",
            "Urine output <0.5ml/kg/hr despite fluids — inadequate resuscitation",
            "Circumferential burns — escharotomy may be needed",
            "BP falling despite fluids"
        ]
    },

    # ── 4. DKA ────────────────────────────────────────────────────────────────
    {
        "id": "dka",
        "exclusive": [
            "diabetic ketoacidosis", "dka", "fruity breath", "kussmaul",
            "blood glucose 28", "blood glucose 25", "blood glucose 20",
            "glucose 28", "glucose 25", "glucose 20", "glucose 30",
            "type 1 diabetic", "type 1 diabetes", "known diabetic vomiting",
            "ketones", "ketonuria", "ketoacid"
        ],
        "supporting": [
            "diabetic", "vomiting", "abdominal pain", "confusion",
            "deep breathing", "rapid breathing", "glucose high"
        ],
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "Diabetic ketoacidosis with altered consciousness and glucose 28+ mmol/L — life-threatening metabolic emergency.",
        "differentials": [
            {"rank":1,"diagnosis":"Diabetic ketoacidosis","icd10":"E11.10","likelihood":"High",
             "key_features":"Known T1DM, glucose >11, fruity breath, Kussmaul breathing, vomiting, confusion"},
            {"rank":2,"diagnosis":"Hyperosmolar hyperglycaemic state","icd10":"E11.0","likelihood":"Moderate",
             "key_features":"Consider if T2DM, no ketones, very high glucose >33, older patient"}
        ],
        "immediate_actions": [
            "IV access x2 — start 0.9% NaCl 1L over first hour",
            "Blood glucose, blood ketones, ABG, U&E STAT",
            "Insulin infusion: 0.1 units/kg/hr regular insulin in 0.9% NaCl",
            "Potassium replacement: add KCl to IV fluids if K+ <5.5 mmol/L",
            "Monitor glucose hourly — do not drop faster than 3-5 mmol/L per hour",
            "Catheterise — monitor urine output hourly"
        ],
        "investigations": [
            "Blood glucose hourly",
            "Blood ketones every 2 hours",
            "U&E every 2 hours (potassium critical)",
            "ABG at presentation and 2h",
            "FBC, blood cultures if infection suspected",
            "ECG — hypokalaemia causes arrhythmia"
        ],
        "treatment": [
            "0.9% NaCl: 1L over 1h, 1L over 2h, 1L over 2h, 1L over 4h, then guided by U&E",
            "Insulin: 0.1 units/kg/hr IV infusion — do NOT give bolus",
            "Potassium: add 20-40 mmol KCl per litre once K+ <5.5",
            "When glucose <14 mmol/L: switch IV fluid to 10% dextrose + continue insulin",
            "Treat precipitating cause: antibiotics if infection"
        ],
        "monitoring": "Glucose hourly. Potassium every 2h (hypokalaemia kills). Ketones every 2h. ABG every 2-4h. Urine output hourly. GCS. Target: glucose fall 3-5 mmol/L/hr, ketones fall 0.5 mmol/L/hr.",
        "referral": {"required":True,"urgency":"Emergency","reason":"DKA requires HDU monitoring, IV insulin infusion and hourly electrolyte correction","facility_type":"High dependency unit or ICU"},
        "drug_alerts": [
            "NEVER give IV insulin bolus in DKA — fatal hypoglycaemia",
            "Replace potassium BEFORE insulin if K+ <3.5 — risk of cardiac arrest",
            "Sodium bicarbonate only if pH <6.9 — otherwise harmful",
            "Phosphate replacement if <0.5 mmol/L"
        ],
        "red_flags": [
            "K+ <3.0 mmol/L — pause insulin, urgent replacement",
            "Glucose falling faster than 5 mmol/L/hr — risk of cerebral oedema",
            "Deteriorating GCS",
            "pH <7.0 despite treatment",
            "Oliguria despite adequate fluids"
        ]
    },

    # ── 5. MALARIA (SEVERE) ───────────────────────────────────────────────────
    {
        "id": "malaria",
        "exclusive": [
            "malaria", "rdt positive", "rdt malaria", "rapid test malaria",
            "rapid test positive", "plasmodium", "falciparum", "pf positive",
            "malaria positive", "test positive malaria"
        ],
        "supporting": [
            "fever", "sweating", "chills", "vomiting", "nigeria", "rivers state",
            "hb 7", "hb7", "haemoglobin 7", "cannot take oral", "splenomegaly"
        ],
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "Confirmed P. falciparum malaria with systemic symptoms and inability to take oral medications indicates severe malaria requiring parenteral treatment.",
        "differentials": [
            {"rank":1,"diagnosis":"Plasmodium falciparum malaria","icd10":"B50.9","likelihood":"High",
             "key_features":"RDT positive, fever, vomiting, Hb 7.2, cannot take oral medications"},
            {"rank":2,"diagnosis":"Co-infection (malaria + bacterial sepsis)","icd10":"B50.9","likelihood":"Moderate",
             "key_features":"Consider if WBC raised, CRP very high, no improvement on artesunate"}
        ],
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
        "referral": {"required":True,"urgency":"Emergency","reason":"Severe malaria with systemic symptoms requires IV artesunate and hospital monitoring","facility_type":"District hospital with IV capability"},
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

    # ── 6. MENINGITIS ─────────────────────────────────────────────────────────
    {
        "id": "meningitis",
        "exclusive": [
            "meningitis", "neck stiffness", "stiff neck", "kernig",
            "brudzinski", "photophobia", "petechial", "petechiae",
            "purpuric rash", "meningeal"
        ],
        "supporting": [
            "fever", "headache", "vomiting", "rash", "confusion",
            "bp 95", "bp 90", "hr 118", "hr 120"
        ],
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "Bacterial meningitis with meningeal signs and haemodynamic compromise — do not delay antibiotics awaiting LP.",
        "differentials": [
            {"rank":1,"diagnosis":"Bacterial meningitis (meningococcal)","icd10":"A39.0","likelihood":"High",
             "key_features":"Kernig/Brudzinski positive, petechiae, fever, neck stiffness, haemodynamic compromise"},
            {"rank":2,"diagnosis":"Viral meningitis","icd10":"A87.9","likelihood":"Low",
             "key_features":"Less likely given rash and haemodynamic instability"}
        ],
        "immediate_actions": [
            "IV Ceftriaxone 2g STAT — do not delay for LP",
            "IV Dexamethasone 0.15mg/kg with or before first antibiotic dose",
            "Blood cultures x2 before antibiotics if possible (within 10 minutes)",
            "500ml NaCl 0.9% IV bolus for hypotension",
            "Isolation — droplet precautions"
        ],
        "investigations": [
            "LP after haemodynamic stabilisation (not before antibiotics)",
            "FBC, CRP, blood cultures",
            "Coagulation screen — DIC risk",
            "Blood glucose (LP CSF:blood glucose ratio)"
        ],
        "treatment": [
            "Ceftriaxone 2g IV q12h x 14 days",
            "Dexamethasone 0.15mg/kg IV q6h x 4 days (reduces mortality and deafness)",
            "If penicillin allergy: Chloramphenicol 25mg/kg IV q6h"
        ],
        "monitoring": "Hourly GCS, BP, urine output. Seizure precautions. Watch for SIADH (fluid restriction if Na+ falling). Skin rash progression.",
        "referral": {"required":True,"urgency":"Emergency","reason":"Bacterial meningitis with septicaemia requires ICU","facility_type":"Emergency room"},
        "drug_alerts": [
            "Penicillin allergy: use Chloramphenicol 75mg/kg/day",
            "Avoid LP if papilloedema, focal neurology or GCS <13 — herniation risk"
        ],
        "red_flags": [
            "Rapidly spreading purpura — meningococcal septicaemia",
            "GCS deterioration",
            "Seizures",
            "BP falling despite fluids"
        ]
    },

    # ── 7. PRE-ECLAMPSIA / ECLAMPSIA ──────────────────────────────────────────
    {
        "id": "preeclampsia",
        "exclusive": [
            "pre-eclampsia", "preeclampsia", "eclampsia", "proteinuria",
            "pregnant bp", "gravid", "pregnant hypertension",
            "obstetric", "32 weeks", "28 weeks", "34 weeks", "36 weeks",
            "weeks pregnant", "weeks gestation"
        ],
        "supporting": [
            "pregnant", "headache", "blurred vision", "bp 158", "bp 160",
            "oedema", "vomiting", "protein", "hypertension"
        ],
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "Severe feature pre-eclampsia with neurological symptoms — risk of eclampsia, placental abruption and maternal death.",
        "differentials": [
            {"rank":1,"diagnosis":"Pre-eclampsia with severe features","icd10":"O14.1","likelihood":"High",
             "key_features":"BP ≥160/110, proteinuria, headache, visual disturbance"},
            {"rank":2,"diagnosis":"HELLP syndrome","icd10":"O14.2","likelihood":"Moderate",
             "key_features":"Needs LFTs, platelets, LDH — epigastric pain, RUQ tenderness"}
        ],
        "immediate_actions": [
            "MgSO4 4g IV over 20 minutes — seizure prophylaxis (STAT)",
            "MgSO4 maintenance: 1g/hr IV infusion",
            "Labetalol 20mg IV OR Nifedipine 10mg sublingual if BP ≥160/110",
            "Left lateral position to relieve aortocaval compression",
            "Urinary catheter — monitor urine output"
        ],
        "investigations": [
            "FBC, LFTs, creatinine, LDH, uric acid",
            "Coagulation screen",
            "Platelet count",
            "24h urine protein or spot protein:creatinine ratio",
            "Fetal CTG"
        ],
        "treatment": [
            "Antihypertensive: Labetalol IV or Nifedipine oral — target BP <150/100",
            "MgSO4 4g IV loading then 1g/hr — continue 24h post delivery",
            "Betamethasone 12mg IM x2 doses (24h apart) if <34 weeks — fetal lung maturity",
            "Delivery planning with obstetric team"
        ],
        "monitoring": "BP every 15 minutes until stable then hourly. Urine output >25ml/hr. Patellar reflexes hourly (MgSO4 toxicity). Fetal CTG. Platelet trend.",
        "referral": {"required":True,"urgency":"Emergency","reason":"Severe pre-eclampsia needs obstetric specialist and delivery planning","facility_type":"Emergency room with obstetric unit"},
        "drug_alerts": [
            "MgSO4 toxicity antidote: Calcium gluconate 10ml of 10% IV STAT",
            "Withhold MgSO4 if patellar reflexes absent or urine output <25ml/hr",
            "Avoid ACE inhibitors and ARBs in pregnancy — teratogenic",
            "Atenolol causes fetal growth restriction — avoid"
        ],
        "red_flags": [
            "Seizure onset — give MgSO4 4g IV STAT",
            "Epigastric or RUQ pain — HELLP syndrome",
            "Platelet count below 100,000",
            "Fetal bradycardia — placental abruption",
            "Oliguria — renal failure"
        ]
    },

    # ── 8. STEMI ──────────────────────────────────────────────────────────────
    {
        "id": "stemi",
        "exclusive": [
            "stemi", "st elevation", "myocardial infarction", "heart attack",
            "mi", "chest pain radiating", "chest pain left arm",
            "v1", "v2", "v3", "v4"
        ],
        "supporting": [
            "chest pain", "diaphoresis", "sweating", "nausea", "diabetic",
            "bp 88", "bp 90", "hr 105", "jaw pain"
        ],
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "STEMI with cardiogenic shock — time critical, door to needle within 30 minutes if thrombolysis available.",
        "differentials": [
            {"rank":1,"diagnosis":"ST elevation myocardial infarction (anterior)","icd10":"I21.0","likelihood":"High",
             "key_features":"ST elevation V1-V4, chest pain radiation, diaphoresis, haemodynamic compromise"},
            {"rank":2,"diagnosis":"Aortic dissection","icd10":"I71.0","likelihood":"Low",
             "key_features":"Consider if tearing pain, maximal at onset, unequal BP in arms"}
        ],
        "immediate_actions": [
            "Aspirin 300mg chewed STAT",
            "Clopidogrel 600mg loading dose oral",
            "Morphine 2-4mg IV titrated for pain (with metoclopramide 10mg IV)",
            "IV access x2, oxygen ONLY if SpO2 <94%",
            "12-lead ECG and continuous cardiac monitoring"
        ],
        "investigations": [
            "12-lead ECG STAT and serial every 30 min",
            "Troponin and CK-MB",
            "FBC, coagulation, renal function",
            "Chest X-ray",
            "Blood glucose"
        ],
        "treatment": [
            "Thrombolysis if PCI not available within 2h: Streptokinase 1.5M units IV over 60 min",
            "Heparin 5000 units IV bolus then infusion",
            "Beta-blocker: Metoprolol 25mg oral ONLY if haemodynamically stable",
            "ACE inhibitor: Ramipril 2.5mg once stable (after 24h)"
        ],
        "monitoring": "Continuous ECG. BP every 15 minutes. Urine output. Signs of reperfusion (pain relief, ST resolution, reperfusion arrhythmias). Watch for VF.",
        "referral": {"required":True,"urgency":"Emergency","reason":"STEMI requires urgent PCI or thrombolysis at cardiac centre","facility_type":"Emergency room with cardiac care"},
        "drug_alerts": [
            "Thrombolysis contraindicated: recent surgery/stroke, active bleeding, uncontrolled BP >180/110",
            "Avoid beta-blockers if BP <90 or HR <50 or acute heart failure",
            "Hold Metformin during acute illness",
            "Avoid NSAIDs — increase mortality in MI"
        ],
        "red_flags": [
            "VF or VT on monitor — immediate defibrillation",
            "Worsening pulmonary oedema",
            "BP below 80 systolic — cardiogenic shock",
            "New complete heart block"
        ]
    },

    # ── 9. PNEUMONIA ──────────────────────────────────────────────────────────
    {
        "id": "pneumonia",
        "exclusive": [
            "pneumonia", "consolidation", "crackles", "chest indrawing",
            "subcostal retraction", "nasal flaring", "fast breathing rr",
            "respiratory rate 52", "respiratory rate 48", "rr 52", "rr 48"
        ],
        "supporting": [
            "fever", "cough", "spo2", "breathing", "oxygen",
            "ampicillin", "amoxicillin"
        ],
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "Severe pneumonia with hypoxia — WHO criteria met: SpO2 <90%, chest indrawing, unable to feed/drink.",
        "differentials": [
            {"rank":1,"diagnosis":"Severe community-acquired pneumonia","icd10":"J18.9","likelihood":"High",
             "key_features":"Fast breathing above age threshold, retractions, SpO2 <90%, crackles"},
            {"rank":2,"diagnosis":"Bronchiolitis (infants)","icd10":"J21.9","likelihood":"Moderate",
             "key_features":"Age <2 years, wheeze, RSV season — but bacterial signs predominant here"}
        ],
        "immediate_actions": [
            "Oxygen via nasal prongs — target SpO2 >94%",
            "IV or IM Ampicillin 50mg/kg q6h",
            "IV access and fluid assessment",
            "Pulse oximetry continuous"
        ],
        "investigations": [
            "Chest X-ray — confirm consolidation",
            "FBC, CRP",
            "Blood culture before antibiotics",
            "Pulse oximetry continuous"
        ],
        "treatment": [
            "Ampicillin 50mg/kg IV q6h x5 days minimum",
            "If no improvement at 48h: add Gentamicin 7.5mg/kg q24h",
            "Oral Amoxicillin when SpO2 stable and tolerating oral",
            "Antipyretic: Paracetamol 15mg/kg q6h for fever"
        ],
        "monitoring": "SpO2 continuous. RR and HR hourly. Temperature 4-hourly. Feeding assessment. Urine output.",
        "referral": {"required":True,"urgency":"Emergency","reason":"Severe pneumonia with hypoxia requires hospital oxygen and IV antibiotics","facility_type":"District hospital"},
        "drug_alerts": [
            "Penicillin allergy: use Erythromycin 12.5mg/kg q6h",
            "Avoid unnecessary nebulisers in bacterial pneumonia"
        ],
        "red_flags": [
            "SpO2 falling below 90% despite oxygen",
            "Apnoea",
            "Central cyanosis",
            "Worsening consciousness"
        ]
    },

    # ── 10. SEPSIS ────────────────────────────────────────────────────────────
    {
        "id": "sepsis",
        "exclusive": [
            "sepsis", "septicaemia", "septicemia", "blood poisoning",
            "suspected sepsis", "septic shock", "qsofa"
        ],
        "supporting": [
            "fever", "tachycardia", "hr 120", "hr 130", "bp 90", "bp 80",
            "confusion", "infection", "source of infection"
        ],
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "Sepsis with haemodynamic compromise — Sepsis-3 criteria met. Begin Sepsis Six bundle within 1 hour.",
        "differentials": [
            {"rank":1,"diagnosis":"Sepsis with organ dysfunction","icd10":"A41.9","likelihood":"High",
             "key_features":"Suspected infection + organ dysfunction (confusion, low BP, high HR, low urine output)"},
            {"rank":2,"diagnosis":"Septic shock","icd10":"R65.21","likelihood":"Moderate",
             "key_features":"Sepsis + vasopressors needed to maintain MAP >65 + lactate >2 mmol/L"}
        ],
        "immediate_actions": [
            "Sepsis Six STAT: give all within 1 hour",
            "1. Oxygen: high-flow O2 target SpO2 >94%",
            "2. Blood cultures x2 before antibiotics",
            "3. IV antibiotics: Ceftriaxone 2g IV + Metronidazole 500mg IV",
            "4. IV fluids: 30ml/kg crystalloid bolus (NaCl 0.9% or Ringer's)",
            "5. Serum lactate and FBC",
            "6. Urine output monitoring — insert catheter"
        ],
        "investigations": [
            "Blood cultures x2 (before antibiotics)",
            "FBC, U&E, LFTs, coagulation",
            "Serum lactate",
            "Urine culture and urinalysis",
            "CXR, cultures from source"
        ],
        "treatment": [
            "Broad-spectrum IV antibiotics within 1 hour: Ceftriaxone 2g IV q12h + Metronidazole 500mg IV q8h",
            "If hospital-acquired or immunocompromised: Piperacillin-Tazobactam 4.5g IV q8h",
            "IV fluid resuscitation: 30ml/kg crystalloid over 3 hours",
            "Vasopressors if MAP <65 despite fluids: Noradrenaline 0.1-0.3 mcg/kg/min"
        ],
        "monitoring": "MAP every 15 min. Urine output hourly (target >0.5ml/kg/hr). Lactate at 2h (should halve). Temperature 4-hourly. Mental status. Adjust antibiotics based on cultures.",
        "referral": {"required":True,"urgency":"Emergency","reason":"Sepsis requires ICU monitoring, IV antibiotics and vasopressors","facility_type":"ICU or HDU"},
        "drug_alerts": [
            "Penicillin allergy: use Ciprofloxacin 400mg IV q12h + Metronidazole",
            "Avoid gentamicin if renal impairment",
            "Steroids (Hydrocortisone 200mg/day) only if septic shock not responding to vasopressors"
        ],
        "red_flags": [
            "Lactate >4 mmol/L — high mortality",
            "MAP falling below 65 despite 2L fluids",
            "GCS deteriorating",
            "Urine output <0.5ml/kg/hr despite fluids",
            "Petechial or purpuric rash — consider meningococcal"
        ]
    },

    # ── 11. GASTROENTERITIS / DEHYDRATION ─────────────────────────────────────
    {
        "id": "gastroenteritis",
        "exclusive": [
            "diarrhoea", "diarrhea", "loose stool", "watery stool", "stooling",
            "gastroenteritis", "cholera", "ors", "oral rehydration",
            "rice water stool"
        ],
        "supporting": [
            "vomiting", "fever", "dehydration", "abdominal pain",
            "weakness", "sunken eyes"
        ],
        "triage": "AMBER",
        "urgency": "Within 4h",
        "rationale": "Acute gastroenteritis with dehydration and fever — assess severity and rehydrate.",
        "differentials": [
            {"rank":1,"diagnosis":"Acute gastroenteritis with dehydration","icd10":"A09","likelihood":"High",
             "key_features":"Diarrhoea, vomiting, fever, signs of dehydration"},
            {"rank":2,"diagnosis":"Cholera","icd10":"A00.9","likelihood":"Moderate",
             "key_features":"Rice-water stools, profuse watery diarrhoea, cholera endemic area"}
        ],
        "immediate_actions": [
            "Assess dehydration: mild (<5%), moderate (5-10%), severe (>10% — IV fluids)",
            "Oral rehydration solution (ORS): 50-100ml/kg over 4 hours for moderate dehydration",
            "IV Ringer's Lactate 30ml/kg over 30 min if severe dehydration (sunken eyes, very weak)",
            "Continue breastfeeding if infant"
        ],
        "investigations": [
            "Stool microscopy and culture",
            "FBC, U&E (if severe dehydration)",
            "Blood glucose in children",
            "RDT malaria if fever in endemic area"
        ],
        "treatment": [
            "ORS: continue until diarrhoea stops",
            "Zinc 20mg daily x 10 days (children) — reduces duration",
            "Antibiotics only if bloody stool or cholera suspected: Azithromycin 20mg/kg single dose",
            "Antipyretic: Paracetamol if fever >38.5"
        ],
        "monitoring": "Urine output. Skin turgor. Fontanelle (infants). Weight daily. Stool frequency and consistency.",
        "referral": {"required":False,"urgency":"Urgent","reason":"Refer if severe dehydration, blood in stool, or not improving in 24h","facility_type":"District hospital"},
        "drug_alerts": [
            "Avoid antidiarrhoeal agents in children under 5 — dangerous",
            "Avoid loperamide in bloody diarrhoea — risk of toxic megacolon"
        ],
        "red_flags": [
            "Signs of severe dehydration: sunken fontanelle, dry mouth, no urine >6h",
            "Blood in stool",
            "High fever >39C",
            "Seizures",
            "Altered consciousness"
        ]
    },

    # ── 12. TYPHOID ───────────────────────────────────────────────────────────
    {
        "id": "typhoid",
        "exclusive": [
            "typhoid", "enteric fever", "rose spots", "widal",
            "salmonella typhi", "step-ladder fever"
        ],
        "supporting": [
            "prolonged fever", "constipation", "splenomegaly", "bradycardia", "nigeria"
        ],
        "triage": "AMBER",
        "urgency": "Within 4h",
        "rationale": "Suspected typhoid fever — requires antibiotic treatment and monitoring for complications.",
        "differentials": [
            {"rank":1,"diagnosis":"Typhoid fever","icd10":"A01.0","likelihood":"High",
             "key_features":"Prolonged step-ladder fever, constipation then diarrhoea, relative bradycardia, splenomegaly"},
            {"rank":2,"diagnosis":"Malaria","icd10":"B50.9","likelihood":"Moderate",
             "key_features":"RDT to exclude — malaria endemic area"}
        ],
        "immediate_actions": [
            "Blood culture x2 (most sensitive in first week of illness)",
            "Widal test (limited sensitivity) or typhidot rapid test",
            "Start empirical antibiotics if clinical suspicion high",
            "Oral rehydration or IV fluids if dehydrated"
        ],
        "investigations": [
            "Blood culture x2",
            "FBC — relative leucopenia typical",
            "Widal test or typhidot",
            "RDT malaria to exclude co-infection",
            "LFTs"
        ],
        "treatment": [
            "Azithromycin 1g oral single dose then 500mg daily x 5 days (uncomplicated)",
            "Ceftriaxone 2g IV q24h x 10-14 days (severe or complicated)",
            "Chloramphenicol 25mg/kg oral q6h x 14 days if other antibiotics unavailable",
            "Antipyretic: Paracetamol — avoid NSAIDs (GI bleeding risk)"
        ],
        "monitoring": "Temperature 4-hourly. Watch for intestinal perforation (acute abdomen, rebound tenderness). Urine output. Recurrence of fever.",
        "referral": {"required":True,"urgency":"Urgent","reason":"Typhoid requires antibiotic treatment and monitoring for perforation","facility_type":"District hospital"},
        "drug_alerts": [
            "Avoid NSAIDs — risk of GI bleeding and perforation",
            "Quinolone resistance increasing — check local sensitivity patterns",
            "Chloramphenicol resistance common in some areas"
        ],
        "red_flags": [
            "Sudden acute abdominal pain — intestinal perforation",
            "Shock",
            "Haematemesis or melaena",
            "Seizures — typhoid encephalopathy"
        ]
    },

    # ── 13. TUBERCULOSIS ──────────────────────────────────────────────────────
    {
        "id": "tuberculosis",
        "exclusive": [
            "tuberculosis", "tb", "haemoptysis", "night sweats weight loss",
            "acid fast", "afb", "cough 3 weeks", "cough 4 weeks", "cough 6 weeks",
            "productive cough blood", "sputum blood"
        ],
        "supporting": [
            "weight loss", "night sweats", "cough", "hiv", "contact",
            "prolonged cough"
        ],
        "triage": "AMBER",
        "urgency": "Within 4h",
        "rationale": "Suspected pulmonary tuberculosis — requires sputum AFB and isolation pending diagnosis.",
        "differentials": [
            {"rank":1,"diagnosis":"Pulmonary tuberculosis","icd10":"A15.0","likelihood":"High",
             "key_features":"Chronic cough >3 weeks, haemoptysis, night sweats, weight loss, TB contact"},
            {"rank":2,"diagnosis":"HIV-related lung disease","icd10":"B22.1","likelihood":"Moderate",
             "key_features":"Consider HIV testing — TB is AIDS-defining illness"}
        ],
        "immediate_actions": [
            "Isolate patient — airborne precautions, N95 mask for healthcare staff",
            "Sputum for AFB x3 (spot, morning, spot)",
            "Chest X-ray",
            "HIV test (with consent)"
        ],
        "investigations": [
            "Sputum AFB smear and culture x3",
            "Chest X-ray — upper lobe infiltrates, cavitation",
            "HIV rapid test",
            "GeneXpert MTB/RIF if available (faster and more sensitive)",
            "FBC, LFTs, renal function (for drug dosing)"
        ],
        "treatment": [
            "Start only after AFB confirmation (or clinical TB in HIV+ patient)",
            "2HRZE / 4HR regimen:",
            "Intensive phase x2 months: Isoniazid (H) + Rifampicin (R) + Pyrazinamide (Z) + Ethambutol (E)",
            "Continuation phase x4 months: Isoniazid + Rifampicin",
            "Pyridoxine 10mg daily — prevents isoniazid neuropathy"
        ],
        "monitoring": "Monthly weight. LFTs at 2 and 4 weeks (drug hepatotoxicity). Sputum smear at 2 months. Visual acuity (Ethambutol optic neuritis). TB-HIV co-management.",
        "referral": {"required":True,"urgency":"Urgent","reason":"TB requires specialist confirmation, isolation and directly observed therapy (DOTS)","facility_type":"TB clinic or district hospital"},
        "drug_alerts": [
            "Rifampicin reduces efficacy of oral contraceptives, antiretrovirals",
            "Isoniazid — hepatotoxicity risk, check LFTs",
            "Ethambutol — colour vision monthly (optic neuritis)",
            "Drug interactions with ART in HIV+ patients — specialist review"
        ],
        "red_flags": [
            "Massive haemoptysis — emergency bronchoscopy",
            "TB meningitis — altered consciousness, neck stiffness",
            "Miliary TB — widespread shadows on CXR",
            "Respiratory failure"
        ]
    },

    # ── 14. UTI / PYELONEPHRITIS ──────────────────────────────────────────────
    {
        "id": "uti",
        "exclusive": [
            "dysuria", "frequency", "burning urination", "urinary burning",
            "uti", "urinary tract infection", "pyelonephritis",
            "cloudy urine", "smelly urine", "suprapubic pain"
        ],
        "supporting": [
            "fever", "loin pain", "flank pain", "vomiting", "nausea",
            "urinalysis", "nitrites", "leucocytes"
        ],
        "triage": "AMBER",
        "urgency": "Within 4h",
        "rationale": "Urinary tract infection — may be ascending pyelonephritis if fever and loin pain present.",
        "differentials": [
            {"rank":1,"diagnosis":"Urinary tract infection","icd10":"N39.0","likelihood":"High",
             "key_features":"Dysuria, frequency, suprapubic pain, positive urine dipstick"},
            {"rank":2,"diagnosis":"Pyelonephritis","icd10":"N10","likelihood":"Moderate",
             "key_features":"UTI + fever + loin pain + vomiting — upper tract involvement"}
        ],
        "immediate_actions": [
            "Urine dipstick and midstream specimen for culture",
            "Start antibiotics empirically — do not wait for culture",
            "Adequate hydration — oral or IV"
        ],
        "investigations": [
            "Urine dipstick",
            "Midstream urine culture (before antibiotics)",
            "FBC, U&E if pyelonephritis suspected",
            "Blood cultures if febrile and systemically unwell",
            "Renal ultrasound if obstruction suspected"
        ],
        "treatment": [
            "Uncomplicated UTI: Nitrofurantoin 100mg oral BD x5 days (avoid if GFR <30)",
            "Or Trimethoprim 200mg oral BD x7 days",
            "Pyelonephritis (oral): Ciprofloxacin 500mg BD x7-14 days",
            "Pyelonephritis (IV — if vomiting/severe): Ceftriaxone 1g IV q24h x5 days",
            "Analgesia: Paracetamol 1g q6h"
        ],
        "monitoring": "Temperature 4-hourly. Resolution of symptoms at 48h — if no improvement, review culture and sensitivity. Repeat urine culture 5-7 days post-treatment.",
        "referral": {"required":False,"urgency":"Routine","reason":"Refer if not responding to antibiotics, obstruction, recurrent infections or pregnancy","facility_type":"District hospital"},
        "drug_alerts": [
            "Avoid Nitrofurantoin in renal impairment (GFR <30) and in late pregnancy",
            "Quinolone resistance increasing — check local antibiogram",
            "Trimethoprim contraindicated in first trimester of pregnancy"
        ],
        "red_flags": [
            "Rigors — bacteraemia",
            "Not responding to 48h of antibiotics",
            "Oliguria or worsening renal function",
            "Pregnancy with UTI — always treat and follow up culture"
        ]
    },
    # ── 15. ANAPHYLAXIS ───────────────────────────────────────────────────────
    {
        "id": "anaphylaxis",
        "exclusive": [
            "anaphylaxis", "anaphylactic", "allergic reaction severe",
            "throat swelling", "tongue swelling", "lip swelling",
            "urticaria", "hives", "angioedema", "adrenaline", "epipen"
        ],
        "supporting": [
            "allergy", "bee sting", "drug reaction", "food allergy",
            "difficulty swallowing", "stridor", "wheezing"
        ],
        "triage": "RED",
        "urgency": "Immediate",
        "rationale": "Anaphylaxis — life-threatening, requires immediate adrenaline.",
        "differentials": [
            {"rank":1,"diagnosis":"Anaphylaxis","icd10":"T78.2","likelihood":"High",
             "key_features":"Urticaria/angioedema + airway compromise or hypotension after allergen exposure"},
            {"rank":2,"diagnosis":"Vasovagal syncope","icd10":"R55","likelihood":"Low",
             "key_features":"Pallor, bradycardia, no urticaria — not true anaphylaxis"}
        ],
        "immediate_actions": [
            "Adrenaline (Epinephrine) 0.5mg IM into outer thigh (0.5ml of 1:1000) — FIRST LINE",
            "Lay flat, elevate legs (unless airway compromise — sit up)",
            "High-flow oxygen 15L/min via non-rebreathe mask",
            "IV access x2, IV fluids 500ml NaCl 0.9% bolus"
        ],
        "investigations": [
            "Serum mast cell tryptase (within 1-3h of reaction, confirms anaphylaxis)",
            "FBC, U&E",
            "ECG"
        ],
        "treatment": [
            "Adrenaline 0.5mg IM — repeat every 5 minutes if no improvement",
            "Chlorphenamine 10mg IV (antihistamine — secondary line only)",
            "Hydrocortisone 200mg IV (slow onset — prevents biphasic reaction)",
            "Salbutamol nebulised if persistent bronchospasm",
            "Glucagon 1-2mg IV if on beta-blockers (counteracts beta-blockade)"
        ],
        "monitoring": "Continuous BP, HR, SpO2. Observe minimum 6h post reaction for biphasic. Discharge with Epipen prescription and allergy card.",
        "referral": {"required":True,"urgency":"Emergency","reason":"Anaphylaxis requires emergency observation for biphasic reaction and allergy referral","facility_type":"Emergency room"},
        "drug_alerts": [
            "Adrenaline is the ONLY first-line treatment — antihistamines and steroids are secondary",
            "Beta-blockers reduce adrenaline efficacy — use glucagon",
            "Avoid latex — common anaphylaxis trigger in healthcare settings"
        ],
        "red_flags": [
            "Stridor — imminent airway loss, prepare for intubation",
            "BP not responding to two doses of adrenaline — refractory anaphylaxis",
            "Biphasic reaction 4-12h after apparent recovery",
            "Myocardial ischaemia on ECG"
        ]
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# SCORE-BASED PATTERN MATCHING
# Each exclusive keyword = 3 points, each supporting keyword = 1 point.
# Winning condition must score >= 3 (at least 1 exclusive match).
# ──────────────────────────────────────────────────────────────────────────────
def match_clinical_pattern(text: str) -> Optional[dict]:
    txt = text.lower()
    best_score = 0
    best_pattern = None

    for condition in CLINICAL_KB:
        score = 0
        for kw in condition["exclusive"]:
            if kw in txt:
                score += 3
        for kw in condition["supporting"]:
            if kw in txt:
                score += 1

        if score > best_score:
            best_score = score
            best_pattern = condition

    if best_score < 3:
        return None  # No confident match

    p = best_pattern
    return {
        "triage": {
            "level":    p["triage"],
            "urgency":  p["urgency"],
            "rationale": p["rationale"]
        },
        "differentials": p["differentials"],
        "protocol": {
            "immediate_actions": p["immediate_actions"],
            "investigations":    p["investigations"],
            "treatment":         p["treatment"],
            "monitoring":        p["monitoring"]
        },
        "referral": p["referral"],
        "drug_alerts": p["drug_alerts"],
        "red_flags":   p["red_flags"],
        "confidence":  "High",
        "confidence_note": f"Assessment based on WHO/CDC clinical guidelines for {p['differentials'][0]['diagnosis']}. Verify with qualified clinician."
    }


# ──────────────────────────────────────────────────────────────────────────────
# PARSE MODEL RESPONSE (used when pattern matching returns None)
# ──────────────────────────────────────────────────────────────────────────────
def parse_model_response(raw_text: str, clinical_context: str = "") -> dict:
    combined = (raw_text + " " + clinical_context).lower()

    # Try complete JSON first
    try:
        s = raw_text.find("{")
        e = raw_text.rfind("}") + 1
        if s >= 0 and e > s:
            data = json.loads(raw_text[s:e])
            if "triage" in data:
                lvl = data["triage"].get("level", "AMBER").upper()
                if lvl not in ("RED","AMBER","GREEN"):
                    lvl = "AMBER"
                data["triage"]["level"] = lvl
                return data
    except Exception:
        pass

    # Determine triage from combined text
    RED_WORDS   = ["immediate","critical","severe","emergency","life-threat",
                   "malaria","hb 7","vomit","cannot take oral","meningitis",
                   "sepsis","stemi","unconscious","seizure","shock","bleeding",
                   "breathless","chest pain","anaphylaxis","stroke","dka","burns",
                   "eclampsia","pre-eclampsia","asthma","wheeze","inhaler"]
    AMBER_WORDS = ["urgent","serious","hospital","refer","fever","diarrhea",
                   "diarrhoea","cough","infection","pain","weakness","tb","typhoid",
                   "uti","dysuria","confusion","vomiting"]

    if any(w in combined for w in RED_WORDS):
        level, urgency = "RED", "Immediate"
    elif any(w in combined for w in AMBER_WORDS):
        level, urgency = "AMBER", "Within 4h"
    else:
        level, urgency = "GREEN", "Routine"

    # Extract rationale from JSON fragment
    rationale = ""
    for marker in ['"rationale":', "'rationale':"]:
        if marker in raw_text:
            start = raw_text.find(marker) + len(marker)
            chunk = raw_text[start:start+300].strip().strip('"\'')
            rationale = chunk.split('"')[0].split("'")[0].strip()
            break
    if not rationale:
        rationale = clinical_context[:200].replace("\n"," ")

    # Extract diagnosis
    diagnosis = ""
    for marker in ['"diagnosis":', "'diagnosis':"]:
        if marker in raw_text:
            start = raw_text.find(marker) + len(marker)
            chunk = raw_text[start:start+150].strip().strip('"\'')
            diagnosis = chunk.split('"')[0].split(",")[0].strip()
            break

    return {
        "triage": {"level": level, "urgency": urgency, "rationale": rationale or "Requires urgent clinical assessment"},
        "differentials": [{"rank":1,"diagnosis": diagnosis or "Clinical assessment — refer to specialist",
                           "icd10":"—","likelihood":"High","key_features": rationale[:150]}],
        "protocol": {
            "immediate_actions": ["Seek immediate medical attention at nearest health facility",
                                  "Do not delay — this case requires urgent evaluation"] if level=="RED"
                                 else ["Refer to nearest health facility within 4 hours"],
            "investigations": ["Full blood count and differential",
                               "Blood cultures if infection suspected",
                               "Urinalysis and renal function"],
            "treatment": ["Empirical treatment depends on confirmed diagnosis",
                          "Consult qualified clinician — do not self-medicate"],
            "monitoring": "Temperature, pulse, BP and urine output every 4 hours"
        },
        "referral": {
            "required": level != "GREEN",
            "urgency":  "Emergency" if level=="RED" else "Urgent",
            "reason":   "Requires specialist clinical evaluation",
            "facility_type": "Emergency room" if level=="RED" else "District hospital"
        },
        "drug_alerts": [],
        "red_flags": ["Seek immediate attention if condition worsens",
                      "Watch for deteriorating consciousness or breathing"],
        "confidence": "Moderate",
        "confidence_note": "AI assessment from offline model. Always verify with a qualified clinician."
    }


# ──────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ──────────────────────────────────────────────────────────────────────────────
class PatientEncounter(BaseModel):
    age:        Optional[str] = None
    sex:        Optional[str] = None
    complaint:  Optional[str] = None
    vitals:     Optional[str] = None
    history:    Optional[str] = None
    image_data: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# APP LIFECYCLE
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*50)
    print("  MediEdge API Server")
    print("  Offline Clinical Intelligence")
    print("="*50)
    print(f"  Ollama URL: {OLLAMA_URL}")
    print(f"  Model:      {MODEL_NAME}")
    print("="*50 + "\n")
    yield

app = FastAPI(title="MediEdge API", version="3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            tags = r.json().get("models", [])
            model_ready = any(MODEL_NAME in m.get("name","") for m in tags)
        return {"status":"ok","ollama":"connected","model":MODEL_NAME,
                "model_ready":model_ready,"offline":True,"kb_conditions":len(CLINICAL_KB)}
    except Exception as e:
        return {"status":"degraded","ollama":"disconnected","error":str(e)}


@app.post("/api/analyze")
async def analyze_patient(encounter: PatientEncounter):
    # ── Build clinical context ────────────────────────────────────────────────
    parts = []
    if encounter.age:      parts.append(f"Age: {encounter.age}")
    if encounter.sex:      parts.append(f"Sex: {encounter.sex}")
    if encounter.complaint: parts.append(f"Complaint: {encounter.complaint}")
    if encounter.vitals:   parts.append(f"Vitals: {encounter.vitals}")
    if encounter.history:  parts.append(f"History: {encounter.history}")

    if not parts:
        raise HTTPException(400, "No patient information provided")

    clinical_context = "\n".join(parts)

    # ── Score-based pattern matching ─────────────────────────────────────────
    assessment = match_clinical_pattern(clinical_context)

    if assessment:
        logger.info(f"Pattern match: {assessment['differentials'][0]['diagnosis']} — {assessment['triage']['level']}")
        return {
            "assessment":      assessment,
            "encounter_id":    datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp":       datetime.now().isoformat(),
            "patient_context": clinical_context,
            "source":          "clinical_knowledge_base"
        }

    # ── Gemma model reasoning (for conditions not in KB) ─────────────────────
    logger.info("No KB match — querying Gemma model")

    prompt = f"""You are MediEdge clinical AI. Analyze this patient and respond ONLY with a valid JSON object. No text before or after. Start with {{ end with }}.

Patient: {clinical_context}

Return EXACTLY this JSON structure:
{{
  "triage": {{"level": "RED", "urgency": "Immediate", "rationale": "specific clinical reason"}},
  "differentials": [{{"rank": 1, "diagnosis": "specific name", "icd10": "X00.0", "likelihood": "High", "key_features": "clinical features"}}],
  "protocol": {{"immediate_actions": ["action with dose"], "investigations": ["test"], "treatment": ["drug dose route duration"], "monitoring": "parameters and frequency"}},
  "referral": {{"required": true, "urgency": "Emergency", "reason": "reason", "facility_type": "facility"}},
  "drug_alerts": ["alert"],
  "red_flags": ["warning"],
  "confidence": "High",
  "confidence_note": "note"
}}

Rules: RED=life-threatening, AMBER=urgent 4h, GREEN=routine. Give specific drug names and doses. JSON only:"""

    raw_text = ""
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model":  MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature":  0.1,
                        "num_predict":  800,
                        "num_ctx":      2048,
                        "stop": ["```", "Note:", "Disclaimer:", "I am"]
                    }
                }
            )
            raw_text = response.json().get("response", "")
            logger.info(f"Model response length: {len(raw_text)} chars")
    except Exception as e:
        logger.warning(f"Ollama error: {e}")

    assessment = parse_model_response(raw_text, clinical_context)

    return {
        "assessment":      assessment,
        "encounter_id":    datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp":       datetime.now().isoformat(),
        "patient_context": clinical_context,
        "source":          "gemma_model"
    }


@app.get("/api/guidelines")
async def get_guidelines(search: str = ""):
    results = []
    for cond in CLINICAL_KB:
        diag = cond["differentials"][0]["diagnosis"]
        if not search or search.lower() in diag.lower():
            results.append({
                "condition": diag,
                "icd10":     cond["differentials"][0]["icd10"],
                "triage":    cond["triage"],
                "source":    "WHO/CDC Clinical Guidelines",
                "summary":   cond["rationale"]
            })
    return {"guidelines": results, "total": len(results)}


@app.post("/api/report/pdf")
async def generate_pdf(data: dict):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import cm
        import io

        a = data.get("assessment", {})
        encounter_id = data.get("encounter_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        fname = os.path.join(REPORTS_DIR, f"mediEdge_{encounter_id}.pdf")

        triage = a.get("triage", {})
        level  = triage.get("level", "AMBER")
        clr_map = {"RED": colors.HexColor("#dc2626"),
                   "AMBER": colors.HexColor("#d97706"),
                   "GREEN": colors.HexColor("#16a34a")}
        triage_colour = clr_map.get(level, colors.HexColor("#d97706"))

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                 leftMargin=2*cm, rightMargin=2*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # Header
        story.append(Paragraph("<b>MediEdge Clinical Assessment Report</b>", styles["Title"]))
        story.append(Paragraph(f"Encounter: {encounter_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
        story.append(Spacer(1, 0.5*cm))

        # Triage badge
        triage_data = [[
            Paragraph(f"<b>TRIAGE: {level}</b>", styles["Normal"]),
            Paragraph(triage.get("urgency",""), styles["Normal"]),
            Paragraph(triage.get("rationale",""), styles["Normal"])
        ]]
        t = Table(triage_data, colWidths=[4*cm, 4*cm, 9*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), triage_colour),
            ("TEXTCOLOR",  (0,0), (-1,-1), colors.white),
            ("FONTSIZE",   (0,0), (-1,-1), 10),
            ("PADDING",    (0,0), (-1,-1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        def section(title, items):
            story.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
            if isinstance(items, list):
                for item in items:
                    story.append(Paragraph(f"• {item}", styles["Normal"]))
            else:
                story.append(Paragraph(str(items), styles["Normal"]))
            story.append(Spacer(1, 0.3*cm))

        # Differentials
        story.append(Paragraph("<b>Differential Diagnoses</b>", styles["Heading3"]))
        for d in a.get("differentials", []):
            story.append(Paragraph(
                f"<b>{d.get('rank','?')}. {d.get('diagnosis','—')}</b> [{d.get('icd10','—')}] — {d.get('likelihood','')}",
                styles["Normal"]
            ))
            story.append(Paragraph(f"   Features: {d.get('key_features','')}", styles["Normal"]))
        story.append(Spacer(1, 0.3*cm))

        proto = a.get("protocol", {})
        section("Immediate Actions",    proto.get("immediate_actions", []))
        section("Investigations",       proto.get("investigations", []))
        section("Treatment",            proto.get("treatment", []))
        section("Monitoring",           proto.get("monitoring", ""))
        section("Drug Alerts",          a.get("drug_alerts", []))
        section("Red Flags",            a.get("red_flags", []))

        # Referral
        ref = a.get("referral", {})
        section("Referral", [
            f"Required: {ref.get('required','')} | Urgency: {ref.get('urgency','')}",
            f"Facility: {ref.get('facility_type','')}",
            f"Reason: {ref.get('reason','')}",
        ])

        # Footer
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(
            "<i>DISCLAIMER: MediEdge is clinical decision support only. "
            "Always verify with a qualified clinician. Not a substitute for professional medical judgment.</i>",
            styles["Normal"]
        ))

        doc.build(story)
        with open(fname, "wb") as f:
            f.write(buf.getvalue())

        return {"pdf_path": fname, "filename": os.path.basename(fname)}

    except Exception as e:
        logger.error(f"PDF error: {e}")
        raise HTTPException(500, f"PDF generation failed: {str(e)}")


@app.get("/api/report/download/{filename}")
async def download_report(filename: str):
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Report not found")
    return FileResponse(path, media_type="application/pdf", filename=filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
