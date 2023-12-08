import pandas as pd
import os
import json
import pickle
from query_language.data_types import *
from query_language.evaluator import TrajectoryDataset
import xgboost
from sklearn.metrics import r2_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

MICROORGANISMS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "microorganism_categories.csv")
DRUG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "drug_categories.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CACHE_DIR = "data/variable_cache"

microorganisms = pd.read_csv(MICROORGANISMS_PATH, index_col=0).drop(columns=["Categorized"])
drug_categories = pd.read_csv(DRUG_PATH, index_col=0)

MICROORGANISMS = {}
for col in microorganisms.columns:
    matching_names = microorganisms[microorganisms[col]].index
    assert not any("," in n for n in matching_names.values)
    MICROORGANISMS[col] = matching_names.values.tolist()
    
PRESCRIPTIONS = {}
for col in drug_categories.columns:
    if col in ("Cardiac Arrest", "all_classes", "Filled") or col.startswith("Unnamed"): continue
    matching_names = drug_categories[drug_categories[col]].index
    assert not any("," in n for n in matching_names.values)
    PRESCRIPTIONS[col] = matching_names.values.tolist()

def load_raw_data(sample=False):
    attributes = AttributeSet(pd.read_feather("data/attributes.arrow"))
    events = EventSet(pd.read_feather("data/events.arrow"), id_field="icustayid", time_field="charttime")
    intervals = IntervalSet(pd.read_feather("data/intervals.arrow"), id_field="icustayid")
    
    los = attributes.get("outtime") - attributes.get("intime")
    valid_patients = los.filter((los >= Duration(4, 'hr')) & (los <= Duration(30, 'days'))).get_ids()
    if sample:
        np.random.seed(1234)
        valid_patients = np.random.choice(valid_patients, size=5000, replace=False)
    attributes = attributes.filter(los.get_ids().isin(valid_patients))
    events = events.filter(events.get_ids().isin(valid_patients))
    intervals = intervals.filter(intervals.get_ids().isin(valid_patients))

    macros = {p: ', '.join(PRESCRIPTIONS[p]) for p in PRESCRIPTIONS}
    print(macros.keys())
    dataset = TrajectoryDataset(attributes, events, intervals, cache_dir=CACHE_DIR, eventtype_macros=macros)
    dataset.query_evaluator.verbose = True
    
    if not os.path.exists("data/train_test_split.pkl"):
        train_patients, val_patients = train_test_split(attributes.index.unique(), train_size=0.333)
        val_patients, test_patients = train_test_split(val_patients, test_size=0.5)
        with open("data/train_test_split.pkl", "wb") as file:
            pickle.dump((train_patients, val_patients, test_patients), file)
    else:
        with open("data/train_test_split.pkl", "rb") as file:
            train_patients, val_patients, test_patients = pickle.load(file)

    return dataset, (train_patients, val_patients, test_patients)

def make_query(variable_definitions, timestep_definition):
    """
    Constructs a query. variable_definitions should be a dictionary mapping
    variable names to dictionaries containing a "query" key. patient_cohort
    and timestep_definition should be strings.
    """
    variable_queries = ',\n\t'.join(f"{name}: {info['query']}" 
                                    for name, info in variable_definitions.items() 
                                    if info["enabled"])
    return f"""
    (
        {variable_queries}
    )
    {timestep_definition}
    """
    
def make_modeling_variables(dataset, variable_definitions, timestep_definition):
    """Creates the variables dataframe."""
    query = make_query(variable_definitions, timestep_definition)
    print(query)
    modeling_variables = dataset.query(query)
    modeling_df = modeling_variables.values

    print("Before:", modeling_df.shape)
    modeling_df = pd.get_dummies(modeling_df, 
                                 columns=[c for c in modeling_df.columns 
                                          if pd.api.types.is_object_dtype(modeling_df[c].dtype) 
                                          or isinstance(modeling_df[c].dtype, pd.CategoricalDtype)])
    print("After:", modeling_df.shape)

    del modeling_variables
    return modeling_df
    
def _train_model(variables, outcomes, train_mask, val_mask, regressor=False, columns_to_drop=None, columns_to_add=None, row_mask=None, **model_params):
    """
    variables: a dataframe containing variables for all patients
    """
    variables = variables.drop(columns=[c for c in variables.columns
                                        if columns_to_drop is not None and re.search(columns_to_drop, c) is not None])
    if row_mask is None: row_mask = np.ones(len(variables), dtype=bool)
    train_X = variables[train_mask & row_mask].values
    train_y = outcomes[train_mask & row_mask]
    val_X = variables[val_mask & row_mask].values
    val_y = outcomes[val_mask & row_mask]
    if columns_to_add is not None:
        train_X = np.hstack([train_X, columns_to_add[train_mask & row_mask].values])
        val_X = np.hstack([val_X, columns_to_add[val_mask & row_mask].values])
    val_sample = np.random.uniform(size=len(val_X)) < 0.1
    
    print("Training", train_X.shape)
    model_cls = xgboost.XGBRegressor if regressor else xgboost.XGBClassifier
    # Don't do class weights - instead, we can simply choose a better operating point
    # if not regressor:
    #     model_params['scale_pos_weight'] = (len(train_y) - train_y.sum()) / train_y.sum()
    model = model_cls(**model_params)
    model.fit(train_X, train_y, eval_set=[(val_X[val_sample], val_y[val_sample])])
    
    print("Evaluating")
    val_pred = model.predict(val_X) if regressor else model.predict_proba(val_X)[:,1]
    metrics = {}
    if regressor:
        metrics["r2_score"] = float(r2_score(val_y, val_pred))
        bin_edges = np.histogram_bin_edges(np.concatenate([val_y, val_pred]), bins=10)
        metrics["hist"] = {
            "values": np.histogram2d(val_y, val_pred, bins=bin_edges)[0].tolist(),
            "bins": bin_edges.tolist()
        }
        hist, bin_edges = np.histogram((val_pred - val_y), bins=10)
        metrics["difference_hist"] = {
            "values": hist.tolist(),
            "bins": bin_edges.tolist()
        }
    else:
        val_y = val_y.astype(np.uint8)
        if len(val_y.unique()) > 1:
            fpr, tpr, thresholds = roc_curve(val_y, val_pred)
            opt_threshold = thresholds[np.argmax(tpr - fpr)]
            metrics["threshold"] = float(opt_threshold)
            metrics["acc"] = float((val_y == (val_pred >= opt_threshold)).mean())
            metrics["roc_auc"] = float(roc_auc_score(val_y, val_pred))
            conf = confusion_matrix(val_y, (val_pred >= opt_threshold))
            metrics["confusion_matrix"] = conf.tolist()
            tn, fp, fn, tp = conf.ravel()
            metrics["sensitivity"] = float(tp / (tp + fn))
            metrics["specificity"] = float(tn / (tn + fp))
    metrics["n_train"] = len(train_X)
    metrics["n_val"] = len(val_X)
    
    # Return preds and true values in the validation set, putting
    # nans whenever the row shouldn't be considered part of the
    # cohort for this model
    preds = np.empty(len(outcomes))
    preds.fill(np.nan)
    preds[(val_mask & row_mask).values] = val_pred
    return model, metrics, preds[val_mask], np.where(val_mask & row_mask, outcomes, np.nan)[val_mask]

def make_model(dataset, model_meta, train_patients, val_patients, modeling_df=None, save_name=None):
    if modeling_df is None:
        modeling_df = make_modeling_variables(dataset, model_meta["variables"], model_meta["timestep_definition"])
        
    outcome = dataset.query("(" + model_meta['outcome'] + 
                                (f" where {model_meta['cohort']}" if model_meta.get('cohort', '') else '') + ")" + 
                                " " + model_meta["timestep_definition"])
        
    print((~pd.isna(outcome.get_values())).sum())
    
    if "regression" not in model_meta:
        model_meta["regression"] = len(np.unique(outcome.get_values()[~pd.isna(outcome.get_values())])) > 2
        
    train_mask = outcome.get_ids().isin(train_patients)
    val_mask = outcome.get_ids().isin(val_patients)
    
    model, metrics, val_pred, val_true = _train_model(
        modeling_df,
        outcome.get_values(),
        train_mask,
        val_mask,
        row_mask=~pd.isna(outcome.get_values()),
        regressor=model_meta.get("regression", False),
        early_stopping_rounds=3)
    
    if save_name is not None:
        # Save out the metadata
        with open(os.path.join(MODEL_DIR, f"spec_{save_name}.json"), "w") as file:
            json.dump(model_meta, file)
        
        # Save out the metrics    
        with open(os.path.join(MODEL_DIR, f"metrics_{save_name}.json"), "w") as file:
            json.dump(metrics, file)
            
        # Save out the model itself and its predictions
        model.save_model(os.path.join(MODEL_DIR, f"model_{save_name}.json"))
        np.save(os.path.join(MODEL_DIR, f"preds_{save_name}.npy"), np.vstack([val_true, val_pred]).T)
        
    return model, metrics, val_pred, val_true
