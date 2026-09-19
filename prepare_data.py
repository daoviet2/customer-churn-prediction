from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / 'data' / 'raw' / 'Customer-Churn.csv'
PROCESSED_PATH = ROOT / 'data' / 'processed' / 'Customer-Churn-processed.csv'


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f'Không tìm thấy dữ liệu raw: {RAW_PATH}')

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(RAW_PATH)

    processed_df = raw_df.drop(columns=['customerID'], errors='ignore').copy()
    processed_df['Churn'] = processed_df['Churn'].map({'Yes': 1, 'No': 0})
    processed_df['TotalCharges'] = pd.to_numeric(
        processed_df['TotalCharges'],
        errors='coerce',
    )
    categorical_columns = processed_df.select_dtypes(include='object').columns
    processed_df = pd.get_dummies(
        processed_df,
        columns=categorical_columns,
        drop_first=True,
        dtype=int,
    )
    processed_df = processed_df.fillna(processed_df.median(numeric_only=True))
    processed_df.to_csv(PROCESSED_PATH, index=False)

    print(f'Raw shape: {raw_df.shape}')
    print(f'Processed shape: {processed_df.shape}')
    print(f'Processed data saved to: {PROCESSED_PATH}')
    print(f'Missing raw values: {int(raw_df.isna().sum().sum())}')


if __name__ == '__main__':
    main()
