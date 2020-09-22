import os
import os.path
import shutil

if __name__ == "__main__":

    path_wctopoae_swissroll_push_kns1 = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WCTopoAE/SwissRoll/push1/kn_seed1'
    path_to_save_wctopoae_swissroll_push_kns1 = '/Users/simons/polybox/Studium/20FS/MT/plot_selected/WCTopoAE/SwissRoll/push1/kn_seed1/latents'

    path_topoae_swissroll_multiseed = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/TopoAE/SwissRoll/multiseed'
    path_to_save_topoae_swissroll_multiseed = '/Users/simons/polybox/Studium/20FS/MT/plot_selected/TopoAE_symmetric/SwissRoll/multiseed/latents'

    path_topoaewc_swissroll_multiseed = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WCTopoAE/SwissRoll/k1_multiseed_new'
    path_to_save_topoaewc_swissroll_multiseed = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE_ext/k1multiseed/latents'

    path_topoaewc_swissroll_kn_multiseed ='/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WCTopoAE/SwissRoll/kn_multiseed'
    path_to_save_topoaewc_swissroll_kn_multiseed = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE_ext/knmultiseed/latents'

    path_topoaewc_swissroll_kn_multiseed_new = '/Users/simons/MT_data/sync/euler_sync/schsimo/MT/output/WCTopoAE/SwissRoll/kn_multiseed_new'
    path_to_save_topoaewc_swissroll_kn_multiseed_new = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE_ext/knmultiseed/latents_new'

    path_topoaewc_vernorm = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE_ext/ver_nonorm2'
    path_to_save_vernorm = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE_ext/ver_nonorm2/0ver_nonorm2_latents'


    #NEW
    path_apush1 = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCTopoAE_swissroll_apush'
    path_to_save_apush1 = '/Users/simons/MT_data/plots/WCTopoAE/swissroll_apush1_latents'

    path_symmetric1 = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCTopoAE_swissroll_symmetric'
    path_to_save_symmetric1 = '/Users/simons/MT_data/plots/WCTopoAE/swissroll_symmetric1_latents'


    path = path_apush1
    path_to_save = path_to_save_apush1
    print('START!')
    i = 0
    for dirpath, dirnames, filenames in os.walk(path):

        for filename in [f for f in filenames if f.endswith("rain_latent_visualization.pdf")]:
            #print os.path.join(dirpath, filename)

            existing_file = open(os.path.join(dirpath, filename), "r")

            new_file = open(os.path.join(path_to_save, dirpath.split('/')[-1] + '.pdf'), "w")

            #dest_dir = src_dir+"/subfolder"
            #src_file = os.path.join(src_dir, 'test.txt.copy2')

            src_file = os.path.join(dirpath, filename)
            dest_dir = path_to_save

            shutil.copy(src_file, dest_dir)  # copy the file to destination dir

            dst_file = os.path.join(dest_dir, filename)
            new_dst_file_name = os.path.join(dest_dir, dirpath.split('/')[-1] + '.pdf')

            os.rename(dst_file, new_dst_file_name)  # rename
            os.chdir(dest_dir)

            print('Next one...')
            i += 1


    print('Total: {}'.format(i))



