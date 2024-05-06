import escher
import cobra
import numpy as np


def search_metab_by_name(model, met_name:str):
    hits = []
    for met in model.metabolites:
        if met_name in met.name:
            hits.append(met)
    return hits

def search_metab_by_weight(model, weight:float, tolerance:float):
    hits = []
    for met in model.metabolites:
        if "R" in met.formula or "X" in met.formula:
            continue
        if np.isclose(weight, met.formula_weight, atol=tolerance, rtol=0.0):
            hits.append(met)
    return hits