"""Check the submission file for the correct format."""

import sys
from datetime import datetime

import pandas as pd
from loguru import logger


def check_submission(df):
    """Check the submission file for the correct format."""
    EXPECTED_COLS = [
        "forecast_date",
        "target",
        "horizon",
        "q0.025",
        "q0.25",
        "q0.5",
        "q0.75",
        "q0.975",
    ]
    LEN_EXP_COLS = len(EXPECTED_COLS)

    TARGETS = ["bikes", "energy", "no2"]

    TARGET_VALS = dict(
        bikes=[str(i) + " day" for i in range(1, 7)],
        energy=[str(i) + " hour" for i in (36, 40, 44, 60, 64, 68)],
        no2=[str(i) + " hour" for i in (36, 40, 44, 60, 64, 68)],
    )

    TARGET_LEN = dict(
        bikes=len(TARGET_VALS["bikes"]),
        energy=len(TARGET_VALS["energy"]),
        no2=len(TARGET_VALS["no2"]),
    )

    TARGET_PLAUS = dict(bikes=[0, 10000], energy=[0, 200], no2=[0, 300])

    COLS_QUANTILES = ["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]

    logger.info("Start checking...")
    logger.info("---------------------------")
    col_names = df.columns

    logger.info("Checking the Columns...")
    # Check column length
    if len(col_names) != LEN_EXP_COLS:
        logger.info(
            "Dataset contains ", len(col_names), "columns. Required are", LEN_EXP_COLS
        )
        logger.info("Stopping early...")
        sys.exit()

    if set(col_names) != set(EXPECTED_COLS):
        logger.info("Dataset does not contain the required columns (or more).")
        missing_cols = set(EXPECTED_COLS) - set(col_names)
        logger.info("The missing columns are:", missing_cols)
        logger.info("Stopping early...")
        sys.exit()

    for i, col in enumerate(EXPECTED_COLS):
        if col == col_names[i]:
            continue
        else:
            logger.info("Columns not in correct order. Order should be:", EXPECTED_COLS)
            logger.info("Your order is:", col_names.values)
            logger.info("Stopping early...")
            sys.exit()

    # Date Col
    logger.info("Checking type of columns...")
    try:
        df["forecast_date"] = pd.to_datetime(
            df["forecast_date"], format="%Y-%m-%d", errors="raise"
        )
    except (pd.errors.ParserError, ValueError):
        logger.info("Could not parse Date in format YYYY-MM-DD")
        logger.info("Stopping early...")
        sys.exit()

    try:
        df["target"] = df["target"].astype("object", errors="raise")
    except ValueError:
        logger.info("Cannot convert target column to String.")
        logger.info("Stopping early...")
        sys.exit()

    try:
        df["horizon"] = df["horizon"].astype("object", errors="raise")
    except ValueError:
        logger.info("Cannot convert horizon column to String.")
        logger.info("Stopping early...")
        sys.exit()

    for cq in COLS_QUANTILES:
        if pd.to_numeric(df[cq], errors="coerce").isna().any():
            logger.warning(
                "Some elements in",
                cq,
                "column are not numeric. This may be fine if you only submit 2 out of 3 targets.",  # noqa: E501
            )
            logger.info("")
            # logger.info("Stopping early...")
            # sys.exit()

    logger.info("Checking if the Dates make sense...")

    if len(pd.unique(df["forecast_date"])) > 1:
        logger.info("forecast_date needs to be the same in all rows.")
        logger.info("Stopping early...")
        sys.exit()

    if df["forecast_date"][0].date() < datetime.today().date():
        logger.warning("Forecast date should not be in the past.")
        logger.info("")
        # warnings.warn("Forecast date should not be in the past.")

    if df["forecast_date"][0].weekday() != 2:
        logger.warning("Forecast date should be a Wednesday.")
        logger.info("")
        # warnings.warn("Forecast date should be a Wednesday")

    logger.info("Checking targets...")

    if not df["target"].isin(TARGETS).all():
        logger.info(f"Target column can only contain {TARGETS}. Check spelling.")
        logger.info("Stopping early...")
        sys.exit()

    for target in TARGETS:
        if len(df[df["target"] == target]) != TARGET_LEN[target]:
            if target == "demand":
                logger.info("Exactly 6 rows need to have target = ", target)
            else:
                logger.info("Exactly 5 rows need to have target =", target)
            logger.info("Stopping early...")
            sys.exit()

        if (df[df["target"] == target]["horizon"] != TARGET_VALS[target]).any():
            logger.info(
                "Target",
                target,
                "horizons need to be (in this order):",
                TARGET_VALS[target],
            )
            logger.info("Stopping early...")
            sys.exit()

        if (df[df["target"] == target][COLS_QUANTILES] < TARGET_PLAUS[target][0]).any(
            axis=None
        ) or (df[df["target"] == target][COLS_QUANTILES] > TARGET_PLAUS[target][1]).any(
            axis=None
        ):
            logger.warning(
                "Implausible values for",
                target,
                "detected. You may want to re-check.",
            )
            logger.info("")
            # warnings.warn("Implausible values for "+str(target)+" detected.
            # You may want to re-check them.")

    logger.info("Checking quantiles...")

    ALL_NAN_IDX = df[df.isna().any(axis=1)].index
    NAN_TARGET_IDX_LIST = []

    if len(ALL_NAN_IDX) != 0:
        NAN_TARGET = df.iloc[ALL_NAN_IDX[0]]["target"]
        #        NAN_TARGET_LENS = dict(DAX = 5,
        #                            energy = 6,
        #                            infections = 5)

        NAN_TARGET_IDX_LIST = df[df["target"] == NAN_TARGET].index

        logger.info(
            "Assume that --",
            NAN_TARGET,
            "-- is your NaN-target. Please DOUBLECHECK if this is correct.",
        )

        #        if len(ALL_NAN_IDX) > NAN_TARGET_LENS[NAN_TARGET]:
        if len(ALL_NAN_IDX) > TARGET_LEN[NAN_TARGET]:
            logger.info(
                "Your dataframe contains more NaNs than entries for target",
                NAN_TARGET,
                ".",
            )
            logger.info("Stopping early...")
            sys.exit()
    else:
        logger.info("Seems like you submitted all three targets. Good job!")

    for i, row in df.iterrows():
        if i in NAN_TARGET_IDX_LIST:
            continue

        diffs = row[COLS_QUANTILES].diff()
        if diffs[1:].isna().any():
            logger.info("Something is wrong with your quantiles.")
            logger.info("Stopping early...")
            sys.exit()
        diffs.iloc[0] = 0
        if (diffs < 0).any():
            logger.info(
                "Predictive quantiles in row",
                i,
                "are not ordered correctly (need to be non-decreasing)",
            )
            logger.info("Stopping early...")
            sys.exit()

    logger.info("---------------------------")
    logger.info("Looks good!")
