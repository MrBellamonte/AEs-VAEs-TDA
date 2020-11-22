from scripts.ssc.wc_offline.config_libraries.local.unity import unity_xytrans1
from src.data_preprocessing.witness_complex_offline.compute_wc import compute_wc_multiple

if __name__ == "__main__":
    compute_wc_multiple(unity_xytrans1)