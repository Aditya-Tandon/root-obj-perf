#!/usr/bin/env python3
"""
Quick test of event_features module on a small sample.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from event_features import pair_jets_to_higgs, extract_hh_features
    from data_pipeline.root_loading import load_and_prepare_data
    import awkward as ak
    import vector

    print("✓ Imports successful")

    # Test with minimal data
    print("\n=== Testing pair_jets_to_higgs ===")

    # Create mock jets
    ak.behavior.update(vector.backends.awkward.behavior)

    mock_jets = ak.zip(
        {
            "pt": [[100, 80, 60, 50]],
            "eta": [[0.5, -0.3, 1.2, -0.8]],
            "phi": [[0.1, 1.5, -0.5, 2.0]],
            "mass": [[15, 12, 18, 10]],
            "vector": [
                ak.zip(
                    {
                        "pt": [100, 80, 60, 50],
                        "eta": [0.5, -0.3, 1.2, -0.8],
                        "phi": [0.1, 1.5, -0.5, 2.0],
                        "mass": [15, 12, 18, 10],
                    },
                    with_name="Momentum4D",
                )
            ],
        }
    )

    h1, h2 = pair_jets_to_higgs(mock_jets)
    print(f"H1 mass: {h1.mass[0]:.1f} GeV")
    print(f"H2 mass: {h2.mass[0]:.1f} GeV")
    print(f"HH mass: {(h1 + h2).mass[0]:.1f} GeV")

    print("\n✓ pair_jets_to_higgs works!")

    # Test feature extraction
    print("\n=== Testing extract_hh_features ===")

    class MockEvents:
        pass

    events = MockEvents()
    features = extract_hh_features(events, mock_jets)

    print(f"Extracted features: {list(features.keys())}")
    print(f"m_HH: {features['m_hh'][0]:.1f} GeV")
    print(f"pT_HH: {features['pt_hh'][0]:.1f} GeV")

    print("\n✓ extract_hh_features works!")
    print("\n=== ALL TESTS PASSED ===")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
