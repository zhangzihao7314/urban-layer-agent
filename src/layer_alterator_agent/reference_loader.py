import pandas as pd


PREDICTOR_COLUMNS = [
    "TCH",
    "IMD",
    "BH",
    "BSF",
    "SVF",
    "F_AC",
    "F_S",
    "F_M",
    "F_BS",
    "F_G",
    "F_TV",
    "F_W",
]


FRACTION_COLUMNS = [
    "F_AC",
    "F_S",
    "F_M",
    "F_BS",
    "F_G",
    "F_TV",
    "F_W",
]


def load_reference_table(file_path: str) -> pd.DataFrame:
    """
    Load LCZ predictor reference table.

    Supported formats:
    - .csv
    - .xlsx
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported reference table format. Use .csv or .xlsx")

    df = add_missing_water_fraction(df)
    check_required_predictor_columns(df)

    return df


def add_missing_water_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    If F_W is missing, calculate it as:
    F_W = 1 - sum(other fraction columns)
    """
    df = df.copy()

    if "F_W" not in df.columns:
        other_fraction_columns = [
            "F_AC",
            "F_S",
            "F_M",
            "F_BS",
            "F_G",
            "F_TV",
        ]

        missing = [
            col for col in other_fraction_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Cannot calculate F_W because these columns are missing: {missing}"
            )

        df["F_W"] = 1.0 - df[other_fraction_columns].sum(axis=1)
        df["F_W"] = df["F_W"].clip(lower=0.0)

    return df


def check_required_predictor_columns(df: pd.DataFrame) -> None:
    """
    Check whether all required predictor columns exist.
    """
    missing_columns = [
        col for col in PREDICTOR_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing predictor columns in reference table: {missing_columns}"
        )


def get_predictor_values(
    df: pd.DataFrame,
    urban_type: str,
    urban_type_column: str = "Class Name"
) -> dict:
    """
    Retrieve predictor values for one matched urban type.
    """
    if urban_type_column not in df.columns:
        raise ValueError(
            f"Column '{urban_type_column}' not found in reference table. "
            f"Available columns: {list(df.columns)}"
        )

    matched_rows = df[
        df[urban_type_column].astype(str).str.lower() == urban_type.lower()
    ]

    if matched_rows.empty:
        raise ValueError(f"Urban type not found in reference table: {urban_type}")

    row = matched_rows.iloc[0]

    values = {}
    for col in PREDICTOR_COLUMNS:
        values[col] = float(row[col])

    return values


def validate_predictor_values(values: dict) -> None:
    """
    Validate Layer Alterator C1 constraints:
    - all values in [0, 1]
    - IMD >= BSF
    - fraction sum = 1
    """
    for key, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} value is outside [0,1]: {value}")

    if values["IMD"] < values["BSF"]:
        raise ValueError(
            f"Invalid values: IMD < BSF ({values['IMD']} < {values['BSF']})"
        )

    fraction_sum = sum(values[col] for col in FRACTION_COLUMNS)

    if abs(fraction_sum - 1.0) > 0.01:
        raise ValueError(
            f"Invalid fraction sum: {fraction_sum}. Expected approximately 1.0"
        )