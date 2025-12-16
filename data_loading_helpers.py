import json
import numpy as np
import awkward as ak
import uproot
import vector

ak.behavior.update(vector.backends.awkward.behavior)

def load_and_prepare_data(file_pattern, tree_name, collections_to_load, max_events, correct_pt=True, CONFIG=None):
    """
    Loads the ROOT file, restructures the flat branches into objects,
    and creates 4-vector representations.
    """
    print(f"Loading data from {file_pattern}...")
    if CONFIG is None:
        with open("hh-bbbb-obj-config.json", "r") as config_file:
            CONFIG = json.load(config_file)

    try:
        events = uproot.concatenate(
                f"{file_pattern}:{tree_name}", 
                library="ak",
                entry_stop=max_events 
            )
    except FileNotFoundError:
            print(f"Error: No files found matching '{file_pattern}'. Please update the path.")
            exit()

    print("Reshaping data into nested objects...")
    for prefix in collections_to_load:
        prefixed_fields = [field for field in events.fields if field.startswith(prefix + "_")]
        if not prefixed_fields:
            print(f"Warning: No fields found with prefix '{prefix}_'. Skipping.")
            continue
        field_map = {field.replace(prefix + "_", ""): events[field] for field in prefixed_fields}
        events[prefix] = ak.zip(field_map)

    print("Creating 4-vector objects...")
    for prefix in collections_to_load:
        if prefix in events.fields and "pt" in events[prefix].fields:
            
            # Default to using the raw pt
            pt_field = events[prefix].pt

            # Handle mass:
            if "mass" in events[prefix].fields:
                mass_field = events[prefix].mass
            elif "et" in events[prefix].fields:
                # Calculate L1 mass from et, pt, and eta
                m2 = (events[prefix].et**2 - events[prefix].pt**2) * (np.cosh(events[prefix].eta)**2)
                m2_positive = ak.where(m2 < 0, 0, m2)
                mass_field = np.sqrt(m2_positive)
            else:
                mass_field = ak.zeros_like(pt_field)

            # Apply pT Corrections if this is the offline jet
            if correct_pt:
                if prefix == CONFIG["offline"]["collection_name"]:
                    tagger_name = CONFIG["offline"]["tagger_name"]
                    print(f"Applying pT regression corrections to {prefix} {tagger_name}...")
                    if tagger_name.startswith("btagPNet"):
                        pt_corrected = (
                            events[prefix].pt 
                            * events[prefix].PNetRegPtRawCorr 
                            * events[prefix].PNetRegPtRawCorrNeutrino
                        )
                    elif tagger_name.startswith("btagUParTAK4"):
                        pt_corrected = (
                            events[prefix].pt 
                            * events[prefix].UParTAK4RegPtRawCorr 
                            * events[prefix].UParTAK4RegPtRawCorrNeutrino
                        )
                    else:
                        pt_corrected = events[prefix].pt  # No correction if unknown tagger
                    
                    pt_corrected = (
                        events[prefix].pt 
                        * events[prefix].PNetRegPtRawCorr 
                        * events[prefix].PNetRegPtRawCorrNeutrino
                    )
                    # Scale mass by the same correction factor
                    correction_factor = ak.where(events[prefix].pt > 0, pt_corrected / events[prefix].pt, 1.0)
                    mass_field = mass_field * correction_factor
                    pt_field = pt_corrected

                elif prefix == CONFIG["l1"]["collection_name"] and "ptCorrection" in events[prefix].fields:
                    pt_corrected = events[prefix].pt * events[prefix].ptCorrection
                    correction_factor = ak.where(events[prefix].pt > 0, pt_corrected / events[prefix].pt, 1.0)
                    mass_field = mass_field * correction_factor
                    pt_field = pt_corrected

            # getting the softmaxed scores for the next gen L1 jets
            if prefix == CONFIG["l1"]["collection_name"] and CONFIG["l1"]["collection_name"].endswith("NG"):
                l1_tag_scores = {field: events[prefix][field] for field in events[prefix].fields if field.endswith("Score")}
                for score in l1_tag_scores.keys():
                    events[prefix, score] = l1_tag_scores[score]

                b_v_udscg_score = events[prefix]["bTagScore"] / (events[prefix]["bTagScore"] + events[prefix]["cTagScore"] + events[prefix]["udsTagScore"] + events[prefix]["gTagScore"])
                c_v_b_score = events[prefix]["cTagScore"] / (events[prefix]["cTagScore"] + events[prefix]["bTagScore"])

                events[prefix, "b_v_udscg_score"] = b_v_udscg_score
                events[prefix, "c_v_b_score"] = c_v_b_score

            events[prefix, "vector"] = ak.zip(
                { "pt": pt_field, "eta": events[prefix].eta, "phi": events[prefix].phi, "mass": mass_field, },
                with_name="Momentum4D",
            )
            et_field = np.sqrt(pt_field**2 + mass_field**2) * np.cosh(events[prefix].eta)
            events[prefix, "et"] = et_field

            e_field = np.sqrt((pt_field * np.cosh(events[prefix].eta))**2 + mass_field**2)
            events[prefix, "e"] = e_field
            
    print(f"Loaded and restructured {len(events)} events.")
    return events

def select_gen_b_quarks_from_higgs(events):
    """
    Finds all b-quarks that are direct descendants of a Higgs boson.
    """
    print("Selecting gen-level b-quarks...")
    is_higgs = events.GenPart.pdgId == 25
    higgs_indices = ak.local_index(events.GenPart)[is_higgs]

    is_b = abs(events.GenPart.pdgId) == 5
    b_mother_idx = events.GenPart.genPartIdxMother
    
    b_mother_idx_expanded = b_mother_idx[:, :, None]
    higgs_indices_expanded = higgs_indices[:, None, :]
    
    comparison_b = (b_mother_idx_expanded == higgs_indices_expanded)
    has_higgs_mother_b = ak.any(comparison_b, axis=2)
    
    is_b_from_H = is_b & has_higgs_mother_b
    gen_b_quarks_from_H = events.GenPart[is_b_from_H]

    print(f"Found {ak.sum(ak.num(gen_b_quarks_from_H))} b-quarks from Higgs decays.")
    return gen_b_quarks_from_H
