"""Test submission functionality."""

import marimo

__generated_with = "0.9.11"
app = marimo.App(width="medium")


@app.cell
def __():
    import numpy as np
    import pandas as pd

    from probafcst.utils import check_submission, create_submission

    return check_submission, create_submission, np, pd


@app.cell
def __(create_submission):
    create_submission()
    return


@app.cell
def __(np):
    bikes_preds = np.zeros((6, 5))
    energy_preds = np.zeros((6, 5))
    no2_preds = np.zeros((6, 5))
    return bikes_preds, energy_preds, no2_preds


@app.cell
def __(bikes_preds, create_submission, energy_preds, no2_preds):
    submission = create_submission(
        bikes_preds=bikes_preds, energy_preds=energy_preds, no2_preds=no2_preds
    )
    return (submission,)


@app.cell
def __(check_submission, submission):
    check_submission(submission)
    return


@app.cell
def __(submission):
    submission.to_csv("output/submission.csv", index=False)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
