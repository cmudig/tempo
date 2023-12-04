import pandas as pd
import os
import pickle
from query_language.data_types import *
from model_training import load_raw_data, make_modeling_variables, make_model, MICROORGANISMS, PRESCRIPTIONS
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

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

if __name__ == '__main__':
    dataset, (train_patients, val_patients, _) = load_raw_data()
                
    n_hours = 4 # number of hours back to get most recent data
    period_hours = 1

    variable_spec = {
        "age": {"category": "Demographics", "query": "{age}"},
        "gender_female": {"category": "Demographics", "query": "{gender} = 'Female'"},
        "weight": {"category": "Demographics", "query": "{weight} [impute median]"},
        "height": {"category": "Demographics", "query": "{height} [impute median]"},
        "BMI": {"category": "Demographics", "query": "{weight} / (({height} / 100) * ({height} / 100)) [impute median]"}
    }
    for col in NUMERICAL_COLUMNS:
        variable_spec[f"{col} Missing"] = {"category": "Vitals", "query": f"exists {{{col}}} from #now - {n_hours} h to #now"}
        variable_spec[col] = {"category": "Vitals", "query": f"last {{{col}}} from #now - {n_hours} h to #now [carry 8 hours, impute mean]"}
        variable_spec[f"Delta {col}"] = {"category": "Vitals", "query": f"(mean {{{col}}} from #now - {n_hours} h to #now) - (mean {{{col}}} from #now - {n_hours * 2} h to #now - {n_hours} h) [impute 0]"}
    for col in DISCRETE_EVENT_COLUMNS:
        variable_spec[col] = {"category": "Vitals" if col == "Heart Rhythm" else "Assessments",
                              "query": f"last {{{col}}} from #now - {n_hours} h to #now [carry 8 hours]"}
    for col in FLUID_TYPES:
        variable_spec[col] = {"category": "Fluids", 
                              "query": f"sum amount {{{col}}} from #now - {n_hours} h to #now [impute 0]"}
    for col in OUTPUT_EVENTS:
        variable_spec[col] = {"category": "Fluids", 
                              "query": f"sum {{{col}}} from #now - {n_hours} h to #now [impute 0]"}
    variable_spec["Input Last 24 h"] = {"category": "Fluids",
                                    "query": f"(sum amount {{{', '.join(FLUID_TYPES)}}} from #now - 24 h to #now) + (case when #now - {{intime}} < 24 h then {{inputpreadm}} else 0 end) [impute 0]"}
    variable_spec["Output Last 24 h"] = {"category": "Fluids",
                                     "query": f"(sum {{{', '.join(OUTPUT_EVENTS)}}} from #now - 24 h to #now) + (case when #now - {{intime}} < 24 h then {{uopreadm}} else 0 end) [impute 0]"}
    for col in VASOPRESSOR_TYPES:
        variable_spec[col] = {"category": "Vasopressors",
                              "query": f"integral rate {{{col}}} from #now - {n_hours} h to #now [impute 0]"}
    for col, names in MICROORGANISMS.items():
        names = ",".join('"' + n + '"' for n in names)
        variable_spec[col] = {"category": "Cultures",
                              "query": f"(max ({{Culture}} in [{names}]) from {{intime}} to #now) > 0 [impute 0]"}
    for col in PROCEDURE_TYPES + list(PRESCRIPTIONS.keys()):
        variable_spec[col] = {"category": "Procedures" if col in PROCEDURE_TYPES else "Prescriptions",
                              "query": f"exists {{{col}}} from #now - {n_hours} h to #now [impute 0]"}

    for val in variable_spec.values(): val["enabled"] = True
    
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

    for task_name, model_meta in modeling_tasks.items():
        print(task_name)
        
        make_model(dataset, model_meta, train_patients, val_patients, modeling_df=modeling_df, save_name=task_name)