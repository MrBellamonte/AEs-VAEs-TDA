from src.data_preprocessing.witness_complex_offline.wc_offline_utils import fetch_data

if __name__ == "__main__":
    path_global_register = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/WitnessComplex_offline/wc_global_register.csv'

    dl1, dm1, dl2, dm2 = fetch_data(uid  = 'SwissRoll-bs64-seed1-b5f72fc6', path_global_register = path_global_register)

    print(type(dl1))
    print(type(dm1))
    print(type(dl2))
    print(type(dm2))