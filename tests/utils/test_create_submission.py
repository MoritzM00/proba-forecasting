"""Tests for create_submission module."""

from io import StringIO

import numpy as np
import pandas as pd

from probafcst.utils.create_submission import create_submission


def test_create_submission():
    """Test create_submission_frame function."""
    expected_frame = """
forecast_date,target,horizon,q0.025,q0.25,q0.5,q0.75,q0.975
2024-10-23,bikes,1 day,2496,4676,5552,6444,7408
2024-10-23,bikes,2 day,2107,4596,5532,6294,7658
2024-10-23,bikes,3 day,1992,3357,4068,4826,5828
2024-10-23,bikes,4 day, 873,1568,2178,2528,3335
2024-10-23,bikes,5 day,1886,4614,5376,6193,7222
2024-10-23,bikes,6 day,2270,4786,5678,6274,7463
2024-10-23,energy,36 hour,52887,59429,61920,65836,71181
2024-10-23,energy,40 hour,48102,54604,57213,62027,68767
2024-10-23,energy,44 hour,49212,51915,55298,59425,62749
2024-10-23,energy,60 hour,47175,49336,51944,56183,60777
2024-10-23,energy,64 hour,42864,45934,47995,53107,59378
2024-10-23,energy,68 hour,43070,45698,48347,52965,56667
2024-10-23,no2,36 hour,NA,NA,NA,NA,NA
2024-10-23,no2,40 hour,NA,NA,NA,NA,NA
2024-10-23,no2,44 hour,NA,NA,NA,NA,NA
2024-10-23,no2,60 hour,NA,NA,NA,NA,NA
2024-10-23,no2,64 hour,NA,NA,NA,NA,NA
2024-10-23,no2,68 hour,NA,NA,NA,NA,NA
"""

    expected_frame = pd.read_csv(StringIO(expected_frame), sep=",")

    forecast_date = "2024-10-23"
    bikes_preds = np.array(
        [
            [2496, 4676, 5552, 6444, 7408],
            [2107, 4596, 5532, 6294, 7658],
            [1992, 3357, 4068, 4826, 5828],
            [873, 1568, 2178, 2528, 3335],
            [1886, 4614, 5376, 6193, 7222],
            [2270, 4786, 5678, 6274, 7463],
        ]
    )
    energy_preds = np.array(
        [
            [52887, 59429, 61920, 65836, 71181],
            [48102, 54604, 57213, 62027, 68767],
            [49212, 51915, 55298, 59425, 62749],
            [47175, 49336, 51944, 56183, 60777],
            [42864, 45934, 47995, 53107, 59378],
            [43070, 45698, 48347, 52965, 56667],
        ]
    )
    no2_preds = None

    actual_frame = create_submission(
        forecast_date=forecast_date,
        bikes_preds=bikes_preds,
        energy_preds=energy_preds,
        no2_preds=no2_preds,
    )
    pd.testing.assert_frame_equal(actual_frame, expected_frame)
