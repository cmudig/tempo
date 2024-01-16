import pandas as pd
import os
import re
import json
from query_language.data_types import *
from model_training import load_raw_data, make_modeling_variables, make_model, make_query, MICROORGANISMS, PRESCRIPTIONS
from sklearn.model_selection import train_test_split
import slice_finding as sf

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
SLICES_DIR = os.path.join(os.path.dirname(__file__), "slices")

COMORBIDITY_FIELDS = ['congestive_heart_failure', 'cardiac_arrhythmias',
       'valvular_disease', 'pulmonary_circulation', 'peripheral_vascular',
       'hypertension', 'paralysis', 'other_neurological', 'chronic_pulmonary',
       'diabetes_uncomplicated', 'diabetes_complicated', 'hypothyroidism',
       'renal_failure', 'liver_disease', 'peptic_ulcer', 'aids', 'lymphoma',
       'metastatic_cancer', 'solid_tumor', 'rheumatoid_arthritis',
       'coagulopathy', 'obesity', 'weight_loss', 'fluid_electrolyte',
       'blood_loss_anemia', 'deficiency_anemias', 'alcohol_abuse',
       'drug_abuse', 'psychoses', 'depression']

NUMERICAL_COLUMNS = [
    'GCS Eye Opening', 'GCS Verbal Response', 'GCS Motor Response', 'RASS', 'Heart Rate', 
    'SysBP', 'mAP', 'DiaBP', 'Respiratory Rate', 'Temperature C', 'Potassium', 'Sodium', 
    'Chloride', 'Glucose', 'BUN', 'Creatinine', 'Magnesium', 'Calcium', 'Ionized Ca', 
    'AST', 'ALT', 'Total Bili', 'Direct Bili', 'Total Protein', 'Albumin', 'Hemoglobin', 
    'Hematocrit', 'RBC', 'WBC Count', 'Platelet Count', 'PTT', 'PT', 'INR', 'Arterial pH', 
    'PaO2', 'PaCO2', 'Arterial BE', 'Lactic Acid', 'Bicarbonate', 'End Tidal CO2', 'SpO2', 
    'C Reactive Protein (CRP)', 'Troponin', 'CO2 Calc', 'CVP', 'Tidal Volume', 'Basophils', 
    'Eosinophils', 'Lymphocytes', 'Monocytes', 'Neutrophils', 'RDW-SD', 'O2 Flow', 'FiO2', 
    'PEEP', 'Mean Airway Pressure', 'Minute Volume', 'Peak Inspiratory Pressure', 
    'Plateau Pressure', 'Venous O2 Sat', 'PAPdia', 'PAPmean', 'PAPsys', 'ACT', 'CI',
    'Pain Level', 'Agitation'
]

DISCRETE_EVENT_COLUMNS = [
    'Heart Rhythm', 'O2 Delivery Device', 'Activity', 'Pain Present', 'Pain Type', 'Abdominal Assessment', 
    'Bowel Sounds', 'Urine Appearance', 'Urine Color', 'Skin Color', 'Skin Condition', 'Skin Integrity', 
    'Skin Temperature', 'Pain Cause', 'Pain Location'
]

FLUID_TYPES = ['Isotonic Crystalloid', 'Hypotonic Crystalloid', 'Hypertonic Crystalloid',
        'Isotonic Colloid', 'Blood Products']
OUTPUT_EVENTS = ['Urine', 'Non-Urine Fluid']
VASOPRESSOR_TYPES = ['Norepinephrine', 'Phenylephrine', 'Epinephrine', 'Dopamine', 'Vasopressin']
PROCEDURE_TYPES = ['Invasive Ventilation', 'Non-invasive Ventilation', 'Dialysis', 'Thoracentesis', 'Cardioversion/Defibrillation']

NORMAL_RANGES = {
    # Chart events
    "GCS Eye Opening": { "min": 4, "max": 5 },
    "GCS Verbal Response": { "min": 5, "max": 6 },
    "GCS Motor Response": { "min": 6, "max": 7 },
    "RASS": { "min": -3, "max": 2 },
    "Heart Rate": { "min": 60, "max": 100 },
    "SysBP": { "max": 129 },
    "mAP": { "min": 60, "max": 100 },
    "DiaBP": { "max": 79 },
    "Respiratory Rate": { "min": 12, "max": 20 },
    "Temperature C": { "min": 36.5, "max": 37.2 },

    # Labs
    "Potassium": { "min": 3.4, "max": 5 },
    "Sodium": { "min": 135, "max": 145 },
    "Chloride": { "min": 95, "max": 108 },
    "Glucose": { "min": 70, "max": 110 },
    "BUN": { "min": 8, "max": 25 },
    "Creatinine": {
        "female": { "min": 0.6, "max": 1.8 },
        "male": { "min": 0.8, "max": 2.4 },
    },
    "Magnesium": { "min": 1.7, "max": 2.2 },
    "Calcium": { "min": 8.5, "max": 10.5 },
    "Ionized Ca": { "min": 4.8, "max": 5.6 },
    "AST": { "female": { "min": 9, "max": 25 }, "male": { "min": 10, "max": 40 } },
    "ALT": { "female": { "min": 7, "max": 30 }, "male": { "min": 10, "max": 55 } },
    "Total Bili": { "min": 0, "max": 1 },
    "Direct Bili": { "min": 0, "max": 0.4 },
    "Total Protein": { "min": 6, "max": 8.3 },
    "Albumin": { "min": 3.1, "max": 4.3 },
    "Hemoglobin": { "female": { "min": 12, "max": 16 }, "male": { "min": 13, "max": 18 } },
    "Hematocrit": { "female": { "min": 36, "max": 46 }, "male": { "min": 37, "max": 49 } },
    "RBC": {
        "female": { "min": 3.92, "max": 5.13 },
        "male": { "min": 4.35, "max": 5.65 },
        },
    "WBC Count": { "min": 4.5, "max": 11 },
    "Platelet Count": { "min": 130, "max": 400 },
    "PTT": { "min": 60, "max": 70 },
    "PT": { "min": 11, "max": 13.5 },
    # C_ACT: { "min": 70, "max": 120 },
    "INR": { "min": 0.8, "max": 1.1 },
    "Arterial pH": { "min": 7.35, "max": 7.45 },
    "PaO2": { "min": 75, "max": 100 },
    "PaCO2": { "min": 35, "max": 45 },
    "Arterial BE": { "min": -4, "max": 2 },
    "Lactic Acid": { "min": 0.5, "max": 1.6 },
    "Bicarbonate": { "min": 20, "max": 32 },
    "End Tidal CO2": { "min": 35, "max": 45 },
    # C_SVO2: { "min": 65, "max": 75 },
    "SpO2": { "min": 95 },
    
    "C Reactive Protein (CRP)": { "min": 0, "max": 10 },
    "Troponin": { "max": 0.04 },
    "CO2 Calc": { "min": 23, "max": 29 },
    "CVP": { "min": 0, "max": 10 },
    "Tidal Volume": {
        "female": { "min": 300, "max": 500 },
        "male": { "min": 400, "max": 600 }
    },
    "Basophils": { "min": 0, "max": 1 },
    "Eosinophils": { "min": 1, "max": 4 },
    "Lymphocytes": { "min": 20, "max": 40 },
    "Monocytes": { "min": 2, "max": 8 },
    "Neutrophils": { "min": 40, "max": 60 },
    "RDW-SD": { "min": 40, "max": 55 },
    "O2 Flow": { "min": 40 },
    "FiO2": { "min": 0.4 },
    "Pain Level": { "max": 6, "min": 1 },
    "Agitation": { "max": 3 }
}

def make_modeling_variable_spec(n_hours):
    variable_spec = {
        "age": {"category": "Demographics", "query": "{age}"},
        "gender_female": {"category": "Demographics", "query": "{gender} = 'Female'"},
        "weight": {"category": "Demographics", "query": "{weight} impute median"},
        "height": {"category": "Demographics", "query": "{height} impute median"},
        "BMI": {"category": "Demographics", "query": "{weight} / (({height} / 100) * ({height} / 100)) impute median"}
    }
    for col in COMORBIDITY_FIELDS:
        variable_spec[col] = {"category": "Demographics", "query": f"{{{col}}} impute 0"}
    for col in NUMERICAL_COLUMNS:
        variable_spec[f"{col} Present"] = {"category": "Vitals", "query": f"exists {{{col}}} from #now - {n_hours} h to #now"}
        variable_spec[col] = {"category": "Vitals", "query": f"last {{{col}}} from #now - {n_hours} h to #now carry 8 hours impute mean"}
        variable_spec[f"{col} Delta"] = {"category": "Vitals", "query": f"(mean {{{col}}} from #now - {n_hours} h to #now) - (mean {{{col}}} from #now - {n_hours * 2} h to #now - {n_hours} h) impute 0"}
    for col in DISCRETE_EVENT_COLUMNS:
        variable_spec[col] = {"category": "Vitals" if col == "Heart Rhythm" else "Assessments",
                              "query": f"last {{{col}}} from #now - {n_hours} h to #now carry 8 hours"}
    for col in FLUID_TYPES:
        variable_spec[col] = {"category": "Fluids", 
                              "query": f"sum amount {{{col}}} from #now - {n_hours} h to #now impute 0"}
    for col in OUTPUT_EVENTS:
        variable_spec[col] = {"category": "Fluids", 
                              "query": f"sum {{{col}}} from #now - {n_hours} h to #now impute 0"}
    variable_spec["Input Last 24 h"] = {"category": "Fluids",
                                    "query": f"(sum amount {{{', '.join(FLUID_TYPES)}}} from #now - 24 h to #now) + (case when #now - {{intime}} < 24 h then {{inputpreadm}} else 0 end) impute 0"}
    variable_spec["Output Last 24 h"] = {"category": "Fluids",
                                     "query": f"(sum {{{', '.join(OUTPUT_EVENTS)}}} from #now - 24 h to #now) + (case when #now - {{intime}} < 24 h then {{uopreadm}} else 0 end) impute 0"}
    for col in VASOPRESSOR_TYPES:
        variable_spec[col] = {"category": "Vasopressors",
                              "query": f"integral rate {{{col}}} from #now - {n_hours} h to #now impute 0"}
    for col, names in MICROORGANISMS.items():
        names = ",".join('"' + n + '"' for n in names)
        variable_spec[col] = {"category": "Cultures",
                              "query": f"(max ({{Culture}} in [{names}]) from {{intime}} to #now) > 0 impute 0"}
    for col in PROCEDURE_TYPES + list(PRESCRIPTIONS.keys()):
        variable_spec[col] = {"category": "Procedures" if col in PROCEDURE_TYPES else "Prescriptions",
                              "query": f"exists {{{col}}} from #now - {n_hours} h to #now impute 0"}

    for val in variable_spec.values(): val["enabled"] = True
    
    return variable_spec
    
def make_slicing_variable_spec(n_hours):
    variable_spec = {
        "age": {"category": "Demographics", "query": "case when {age} < 25 then '< 25' when {age} < 45 then '25 - 45' when {age} < 65 then '45 - 65' else '> 65' end impute 'Missing'"},
        "gender_female": {"category": "Demographics", "query": "{gender}"},
        "weight": {"category": "Demographics", "query": "case when {weight} < 50 then '< 50' when {weight} < 100 then '50 - 100' when {weight} < 200 then '100 - 200' else '> 200' end impute 'Missing'"},
        "height": {"category": "Demographics", "query": "case when {height} < 150 then '< 150' when {height} < 180 then '150 - 180' else '> 180' end impute 'Missing'"},
        "BMI": {"category": "Demographics", "query": """
                case when bmi < 18.5 then 'Underweight'
                when bmi < 25 then 'Healthy Range'
                when bmi < 30 then 'Overweight'
                when bmi < 40 then 'Obese'
                else 'Severely Obese' end with bmi as {weight} / ({height} * {height} / 10000) 
                impute 'Missing'
        """}
    }
    for col in COMORBIDITY_FIELDS:
        variable_spec[col] = {"category": "Demographics", "query": f"case when {{{col}}} > 0 then 'Yes' else 'No' end impute 'No'"}
    for col in NUMERICAL_COLUMNS:
        if col not in NORMAL_RANGES:
            print(col, "not included")
            continue
        limits = NORMAL_RANGES[col]
        if "male" in limits:
            variable_spec[col] = {"category": "Vitals", "query": f"""
                case {f'when ({{gender}} = "Male" and last_val < {limits["male"]["min"]}) or ({{gender}} = "Female" and last_val < {limits["female"]["min"]}) then "Low"' if "min" in limits["male"] else ''}
                {f'when ({{gender}} = "Male" and last_val > {limits["male"]["max"]}) or ({{gender}} = "Female" and last_val > {limits["female"]["max"]}) then "High"' if "max" in limits["male"] else ''}
                else "Normal" end
                with last_val as last {{{col}}} from #now - {n_hours} h to #now
                carry 8 hours
                impute "Missing"
            """}
        else:
            variable_spec[col] = {"category": "Vitals", "query": f"""
                case {f'when last_val < {limits["min"]} then "Low"' if "min" in limits else ''}
                {f'when last_val > {limits["max"]} then "High"' if "max" in limits else ''}
                else "Normal" end
                with last_val as last {{{col}}} from #now - {n_hours} h to #now
                carry 8 hours
                impute "Missing"
            """}
        variable_spec[f"Delta {col}"] = {"category": "Vitals", "query": f"""
            case when change < -0.5 then 'Decreasing'
            when change > 0.5 then 'Increasing'
            else 'No Change' end
            with change as (mean {{{col}}} from #now - {n_hours} h to #now) - (mean {{{col}}} from #now - {n_hours * 2} h to #now - {n_hours} h)
            impute 'No Change'
        """}
    for col in DISCRETE_EVENT_COLUMNS:
        variable_spec[col] = {"category": "Vitals" if col == "Heart Rhythm" else "Assessments",
                            "query": f"last {{{col}}} from #now - {n_hours} h to #now carry 8 hours impute 'Missing'"}
    for col in FLUID_TYPES:
        variable_spec[col] = {"category": "Fluids", 
                            "query": f"""
                                case when fluid < 100 then '< 100 mL'
                                when fluid < 500 then '100 - 500 mL'
                                when fluid < 1000 then '500 - 1000 mL'
                                when fluid < 5000 then '1 - 5 L'
                                else '> 5 L' end with fluid as sum amount {{{col}}} from #now - {n_hours} h to #now
                                impute '< 100 mL'
                            """}
    for col in OUTPUT_EVENTS:
        variable_spec[col] = {"category": "Fluids", 
                            "query": f"""
                                case when fluid < 100 then '< 100 mL'
                                when fluid < 500 then '100 - 500 mL'
                                when fluid < 1000 then '500 - 1000 mL'
                                when fluid < 5000 then '1 - 5 L'
                                else '> 5 L' end with fluid as sum {{{col}}} from #now - {n_hours} h to #now
                                impute '< 100 mL'
                            """}
    variable_spec["Input Last 24 h"] = {"category": "Fluids",
                                    "query": f"""
                                    case when fluid < 100 then '< 100 mL'
                                when fluid < 500 then '100 - 500 mL'
                                when fluid < 1000 then '500 - 1000 mL'
                                when fluid < 5000 then '1 - 5 L'
                                else '> 5 L' end 
                                with fluid as ((sum amount {{{', '.join(FLUID_TYPES)}}} from #now - 24 h to #now) + (case when #now - {{intime}} < 24 h then {{inputpreadm}} else 0 end)) 
                                impute '< 100 mL'
                                    """}
    variable_spec["Output Last 24 h"] = {"category": "Fluids",
                                    "query": f"""
                                    case when fluid < 100 then '< 100 mL'
                                when fluid < 500 then '100 - 500 mL'
                                when fluid < 1000 then '500 - 1000 mL'
                                when fluid < 5000 then '1 - 5 L'
                                else '> 5 L' end 
                                with fluid as ((sum {{{', '.join(OUTPUT_EVENTS)}}} from #now - 24 h to #now) + (case when #now - {{intime}} < 24 h then {{uopreadm}} else 0 end)) 
                                impute '< 100 mL'"""}
    for col in VASOPRESSOR_TYPES:
        if col == "Vasopressin":
            variable_spec[col] = {"category": "Vasopressors",
                                "query": f"""
                                case when vaso < 0.1 then 'None'
                                when vaso <= 2 * {n_hours} then '<= 2 units/hour'
                                else '> 2 units/hour' end
                                with vaso as integral rate {{{col}}} from #now - {n_hours} h to #now
                                impute 'None'"""}
        else:
            variable_spec[col] = {"category": "Vasopressors",
                                "query": f"""
                                case when vaso < 10 then 'None'
                                when vaso <= 10000 then '<= 10 mg/kg'
                                else '> 10 mg/kg' end
                                with vaso as integral rate {{{col}}} from #now - {n_hours} h to #now
                                impute 'None'"""}
    for col, names in MICROORGANISMS.items():
        names = ",".join('"' + n + '"' for n in names)
        variable_spec[col] = {"category": "Cultures",
                            "query": f"case when c then 'Yes' else 'No' end with c as (max ({{Culture}} in [{names}]) from {{intime}} to #now) > 0 impute 'No'"}
    for col in PROCEDURE_TYPES + list(PRESCRIPTIONS.keys()):
        variable_spec[col] = {"category": "Procedures" if col in PROCEDURE_TYPES else "Prescriptions",
                            "query": f"case when p then 'Yes' else 'No' end with p as (exists {{{col}}} from #now - {n_hours} h to #now) impute 'No'"}

    for val in variable_spec.values(): 
        val["enabled"] = True
        val["query"] = re.sub(r"\n\s+", "\n", val["query"], flags=re.MULTILINE).strip()

    return variable_spec
    
if __name__ == '__main__':
    dataset, (train_patients, val_patients, _) = load_raw_data()
                
    n_hours = 4 # number of hours back to get most recent data
    period_hours = 1
    
    with open(os.path.join(DATA_DIR, "config.json"), "w") as file:
        json.dump({
            "models": {
                "data_summary": {
                    "fields": [
                        {"name": "Age", "query": "{age}"},
                        {"name": "Gender", "query": "case when {gender} = 1 then 'Female' else 'Male' end"},
                        {"name": "Mortality", "query": "{morta_hosp}"},
                        {"name": "Hours in ICU", "query": "({outtime} - {intime}) / 3600"},
                        {"name": "Most Common Comorbidities", "type": "group", "children": [
                            {"name": col, "query": f"case when {{{col}}} > 0 then 1 else 0 end impute 0"}
                            for col in COMORBIDITY_FIELDS
                        ], "sort": "rate", "ascending": False, "topk": 10},
                    ]
                }
            }
        }, file)

    print("Building modeling variables ========================")
    variable_spec = make_modeling_variable_spec(n_hours)
    timestep_definition = f"every {period_hours} h from {{intime}} + {period_hours} h to {{outtime}}"
    
    modeling_df = make_modeling_variables(dataset, variable_spec, timestep_definition)
        
    modeling_tasks = {
        "vasopressor_8h": {
            "variables": variable_spec,
            "outcome": f"(integral rate {{Norepinephrine, Phenylephrine, Epinephrine, Dopamine, Vasopressin}} from #now to #now + 8 h) > 0.01",
            "cohort": f"({{outtime}} - #now >= 8 hours) and not (exists {{Norepinephrine, Phenylephrine, Epinephrine, Dopamine, Vasopressin}} from {{intime}} to #now)",
            "timestep_definition": timestep_definition,
            "regression": False
        },
        "ventilation_8h": {
            "variables": variable_spec,
            "outcome": f"exists {{Invasive Ventilation, Non-invasive Ventilation}} from #now to #now + 8 h",
            "cohort": f"({{outtime}} - #now >= 8 hours) and not (exists {{Invasive Ventilation, Non-invasive Ventilation}} from {{intime}} to #now)",
            "timestep_definition": timestep_definition,
            "regression": False
        },
        "antimicrobial_8h": {
            "variables": variable_spec,
            "outcome": f"exists {{Antibiotic, Antiviral, Antifungal}} from #now to #now + 8 h",
            "cohort": f"({{outtime}} - #now >= 8 hours) and not (exists {{Antibiotic, Antiviral, Antifungal}} from {{intime}} to #now)",
            "timestep_definition": timestep_definition,
            "regression": False
        }
    }

    print("Building models ========================")
    for task_name, model_meta in modeling_tasks.items():
        print(task_name)
        
        make_model(dataset, model_meta, train_patients, val_patients, modeling_df=modeling_df, save_name=task_name)
        
    # Build the initial slicing specification
    print("Building slicing variables ========================")
    variable_spec = make_slicing_variable_spec(n_hours)
    for var in variable_spec:
        try:
            dataset.parser.parse(variable_spec[var]["query"])
        except Exception as e:
            raise ValueError(f"Exception parsing {var}: {e}")
    
    slice_spec_dir = os.path.join(SLICES_DIR, "specifications")
    if not os.path.exists(slice_spec_dir):
        os.mkdir(slice_spec_dir)
    with open(os.path.join(slice_spec_dir, "default.json"), "w") as file:
        json.dump({
            "variables": variable_spec,
            "slice_filter": sf.filters.ExcludeIfAny([
                sf.filters.ExcludeFeatureValueSet(COMORBIDITY_FIELDS + PROCEDURE_TYPES + list(PRESCRIPTIONS.keys()), ["No"]),
                sf.filters.ExcludeFeatureValueSet(list(variable_spec.keys()), ["Missing"])
            ]).to_dict()
        }, file)
        
