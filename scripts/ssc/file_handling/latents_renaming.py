import os
import os.path
import shutil

if __name__ == "__main__":

    path_wctopoae_swissroll_push_kns1 = '/Users/simons/polybox/Studium/20FS/MT/sync/euler_sync/schsimo/MT/output/WCTopoAE/SwissRoll/push1/kn_seed1'
    path_to_save_wctopoae_swissroll_push_kns1 = '/Users/simons/polybox/Studium/20FS/MT/plot_selected/WCTopoAE/SwissRoll/push1/kn_seed1/latents'

    path_topoae_swissroll_multiseed = '/Users/simons/polybox/Studium/20FS/MT/sync/euler_sync/schsimo/MT/output/TopoAE/SwissRoll/multiseed'
    path_to_save_topoae_swissroll_multiseed = '/Users/simons/polybox/Studium/20FS/MT/plot_selected/TopoAE_symmetric/SwissRoll/multiseed/latents'

    path = path_topoae_swissroll_multiseed
    path_to_save = path_to_save_topoae_swissroll_multiseed
    print('START!')
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



