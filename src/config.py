
TARGET_COL = "channel"
ID_COL = "document_number"
DATE_COL = "registry_date"

TRAIN_START = "2023-01-01"
TRAIN_END   = "2024-09-30"

VAL_START   = "2024-10-01"
VAL_END     = "2024-12-31"

TEST_START  = "2025-01-01"
TEST_END    = "2025-01-31"

NUMERIC_FEATURES = ["year", "month", "weekday"]

CATEGORICAL_FEATURES = [
    "clearance_place_dispatch",
    "clearance_place_entry",
    "consignee_company_size",
    "clearance_place",
    "transport_mode_pt",
    "ncm_code",
    "country_origin_code",
]