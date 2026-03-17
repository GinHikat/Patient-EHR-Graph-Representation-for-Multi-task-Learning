from owlready2 import *
import pandas as pd
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import *

owl = os.path.join(doid_path, 'src', 'ontology', 'doid.owl')

onto = get_ontology(owl).load()
edges = []

def extract_targets(value):
    """
    Recursively extract class names from OWL expressions
    """
    
    # Case 1: simple class
    if hasattr(value, "name") and value.name is not None:
        return [value.name]
    
    # Case 2: logical AND / OR
    elif isinstance(value, (And, Or)):
        results = []
        for v in value.Classes:
            results.extend(extract_targets(v))
        return results
    
    # Case 3: nested restriction
    elif isinstance(value, Restriction):
        return extract_targets(value.value)
    
    return []

def is_valid(name):
    """Filter invalid nodes"""
    return (
        name is not None and
        name not in ["Thing", "Nothing"])

# IS_A RELATIONS
for cls in onto.classes():
    for parent in cls.is_a:
        if isinstance(parent, ThingClass):
            if is_valid(cls.name) and is_valid(parent.name):
                edges.append({
                    "source": cls.name,
                    "relation": "is_a",
                    "target": parent.name
                })

# OBJECT PROPERTY RELATIONS
for prop in onto.object_properties():
    for subj, obj in prop.get_relations():
        if hasattr(subj, "name") and hasattr(obj, "name"):
            if is_valid(subj.name) and is_valid(obj.name):
                edges.append({
                    "source": subj.name,
                    "relation": prop.name.lower(),
                    "target": obj.name
                })

# RESTRICTION-BASED RELATIONS
for cls in onto.classes():
    for restriction in cls.is_a:
        if isinstance(restriction, Restriction):
            if restriction.property and restriction.value:
                
                targets = extract_targets(restriction.value)
                
                for t in targets:
                    if is_valid(cls.name) and is_valid(t):
                        edges.append({
                            "source": cls.name,
                            "relation": restriction.property.name.lower(),
                            "target": t
                        })

edges_df = pd.DataFrame(edges).drop_duplicates()