from src.datasets.datasets import Unity_RotCorgi
from src.evaluation.config import ConfigEval
from src.models.WitnessComplexAE.config import ConfigGrid_WCAE
from src.models.autoencoder.autoencoders import ConvAE_Unity480320

corgi_30_decay = ConfigGrid_WCAE(
        learning_rate=[1/100,1/1000],
        batch_size=[30],
        n_epochs=[5000],
        weight_decay=[1e-6,1e-8,0],
        early_stopping=[50],
        rec_loss_weight=[1],
        top_loss_weight=[1024],
        match_edges=['push_active'],
        k=[2],
        r_max=[10],
        model_class=[ConvAE_Unity480320],
        model_kwargs=[dict()],
        dataset=[Unity_RotCorgi()],
        sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA',version = 5, landmarks = True)],
        eval=[ConfigEval(
            active=True,
            evaluate_on='test',
            eval_manifold=False,
            save_eval_latent=True,
            save_train_latent=True,
            online_visualization=False,
            k_min=5,
            k_max=10,
            k_step=5,
            quant_eval = False
        )],
        uid=[''],
        toposig_kwargs=[dict()],
        method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1], online_wc=[True], dist_x_land = [True],
                         lam_t_decay = [{0: 1024, 25 : 512, 50 : 256, 75: 128, 100 : 64, 125 : 32, 150: 16, 150: 8, 500 : 4, 1000 : 2}],
                         wc_offline=[dict(path_to_data='/cluster/home/schsimo/MT/AEs-VAEs-TDA/src/datasets/simulated/corgi_rotation_5_l')]),
        experiment_dir='/cluster/scratch/schsimo/output/corgi/corgi_30_std',
        seed=1,
        device='cpu',
        num_threads=2,
        verbose=False,
    )


corgi_60_decay = ConfigGrid_WCAE(
        learning_rate=[1/100,1/1000],
        batch_size=[60],
        n_epochs=[5000],
        weight_decay=[1e-6,1e-8,0],
        early_stopping=[50],
        rec_loss_weight=[1],
        top_loss_weight=[1024],
        match_edges=['push_active'],
        k=[2],
        r_max=[10],
        model_class=[ConvAE_Unity480320],
        model_kwargs=[dict()],
        dataset=[Unity_RotCorgi()],
        sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA',version = 6, landmarks = True)],
        eval=[ConfigEval(
            active=True,
            evaluate_on='test',
            eval_manifold=False,
            save_eval_latent=True,
            save_train_latent=True,
            online_visualization=False,
            k_min=5,
            k_max=10,
            k_step=5,
            quant_eval = False
        )],
        uid=[''],
        toposig_kwargs=[dict()],
        method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1], online_wc=[True], dist_x_land = [True],
                         lam_t_decay = [{0: 1024, 25 : 512, 50 : 256, 75: 128, 100 : 64, 125 : 32, 150: 16, 150: 8, 500 : 4, 1000 : 2}],
                         wc_offline=[dict(path_to_data='/cluster/home/schsimo/MT/AEs-VAEs-TDA/src/datasets/simulated/corgi_rotation_6_l')]),
        experiment_dir='/cluster/scratch/schsimo/output/corgi/corgi_60_std',
        seed=1,
        device='cpu',
        num_threads=2,
        verbose=False,
    )

corgi_30_decay_semi = ConfigGrid_WCAE(
        learning_rate=[1/100,1/1000],
        batch_size=[30],
        n_epochs=[5000],
        weight_decay=[1e-6,1e-8,0],
        early_stopping=[50],
        rec_loss_weight=[1],
        top_loss_weight=[8192],
        match_edges=['push_active'],
        k=[2],
        r_max=[10],
        model_class=[ConvAE_Unity480320],
        model_kwargs=[dict()],
        dataset=[Unity_RotCorgi()],
        sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA',version = 5, landmarks = True)],
        eval=[ConfigEval(
            active=True,
            evaluate_on='test',
            eval_manifold=False,
            save_eval_latent=True,
            save_train_latent=True,
            online_visualization=False,
            k_min=5,
            k_max=10,
            k_step=5,
            quant_eval = False
        )],
        uid=[''],
        toposig_kwargs=[dict()],
        method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1], online_wc=[True], dist_x_land = [True],
                         lam_t_bi = [[0,1],[0]], lam_t_decay = [{0: 8192, 50 : 4096, 75 : 2048, 100: 1024, 125 : 512, 150 : 256, 200: 128, 250: 64, 500 : 32, 1000 : 16}],
                         wc_offline=[dict(path_to_data='/cluster/home/schsimo/MT/AEs-VAEs-TDA/src/datasets/simulated/corgi_rotation_5_l')]),
        experiment_dir='/cluster/scratch/schsimo/output/corgi/corgi_30_semi',
        seed=1,
        device='cpu',
        num_threads=2,
        verbose=False,
    )


corgi_60_decay_semi = ConfigGrid_WCAE(
        learning_rate=[1/100,1/1000],
        batch_size=[60],
        n_epochs=[5000],
        weight_decay=[1e-6,1e-8,0],
        early_stopping=[50],
        rec_loss_weight=[1],
        top_loss_weight=[8192],
        match_edges=['push_active'],
        k=[2],
        r_max=[10],
        model_class=[ConvAE_Unity480320],
        model_kwargs=[dict()],
        dataset=[Unity_RotCorgi()],
        sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA',version = 6, landmarks = True)],
        eval=[ConfigEval(
            active=True,
            evaluate_on='test',
            eval_manifold=False,
            save_eval_latent=True,
            save_train_latent=True,
            online_visualization=False,
            k_min=5,
            k_max=10,
            k_step=5,
            quant_eval = False
        )],
        uid=[''],
        toposig_kwargs=[dict()],
        method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1], online_wc=[True], dist_x_land = [True],
                        lam_t_bi = [[0,1],[0]], lam_t_decay = [{0: 8192, 50 : 4096, 75 : 2048, 100: 1024, 125 : 512, 150 : 256, 200: 128, 250: 64, 500 : 32, 1000 : 16}],
                         wc_offline=[dict(path_to_data='/cluster/home/schsimo/MT/AEs-VAEs-TDA/src/datasets/simulated/corgi_rotation_6_l')]),
        experiment_dir='/cluster/scratch/schsimo/output/corgi/corgi_60_semi',
        seed=1,
        device='cpu',
        num_threads=2,
        verbose=False,
    )



corgi_30_decay_semi_long = ConfigGrid_WCAE(
        learning_rate=[1/100,1/1000],
        batch_size=[60],
        n_epochs=[5000],
        weight_decay=[1e-6,1e-8,0],
        early_stopping=[200],
        rec_loss_weight=[1],
        top_loss_weight=[8192],
        match_edges=['push_active'],
        k=[2],
        r_max=[10],
        model_class=[ConvAE_Unity480320],
        model_kwargs=[dict()],
        dataset=[Unity_RotCorgi()],
        sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA',version = 6, landmarks = True)],
        eval=[ConfigEval(
            active=True,
            evaluate_on='test',
            eval_manifold=False,
            save_eval_latent=True,
            save_train_latent=True,
            online_visualization=False,
            k_min=5,
            k_max=10,
            k_step=5,
            quant_eval = False
        )],
        uid=[''],
        toposig_kwargs=[dict()],
        method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1], online_wc=[True], dist_x_land = [True],
                        lam_t_bi = [[0,1]], lam_t_decay = [{0: 8192, 50 : 4096, 75 : 2048, 100: 1024, 125 : 512, 150 : 256, 200: 128, 250: 64, 500 : 32, 1000 : 16}],
                         wc_offline=[dict(path_to_data='/cluster/home/schsimo/MT/AEs-VAEs-TDA/src/datasets/simulated/corgi_rotation_5_l')]),
        experiment_dir='/cluster/scratch/schsimo/output/corgi/corgi_30_semi_long',
        seed=1,
        device='cpu',
        num_threads=3,
        verbose=False,
    )

corgi_30_decay_semi_long_0 = ConfigGrid_WCAE(
        learning_rate=[1/100,1/1000],
        batch_size=[60],
        n_epochs=[5000],
        weight_decay=[1e-6,1e-8,0],
        early_stopping=[200],
        rec_loss_weight=[1],
        top_loss_weight=[8192],
        match_edges=['push_active'],
        k=[2],
        r_max=[10],
        model_class=[ConvAE_Unity480320],
        model_kwargs=[dict()],
        dataset=[Unity_RotCorgi()],
        sampling_kwargs=[dict(root_path = '/cluster/home/schsimo/MT/AEs-VAEs-TDA',version = 6, landmarks = True)],
        eval=[ConfigEval(
            active=True,
            evaluate_on='test',
            eval_manifold=False,
            save_eval_latent=True,
            save_train_latent=True,
            online_visualization=False,
            k_min=5,
            k_max=10,
            k_step=5,
            quant_eval = False
        )],
        uid=[''],
        toposig_kwargs=[dict()],
        method_args=dict(n_jobs=[1], normalize=[True], mu_push=[1], online_wc=[True], dist_x_land = [True],
                        lam_t_bi = [[0]], lam_t_decay = [{0: 8192, 50 : 4096, 75 : 2048, 100: 1024, 125 : 512, 150 : 256, 200: 128, 250: 64, 500 : 32, 1000 : 16}],
                         wc_offline=[dict(path_to_data='/cluster/home/schsimo/MT/AEs-VAEs-TDA/src/datasets/simulated/corgi_rotation_5_l')]),
        experiment_dir='/cluster/scratch/schsimo/output/corgi/corgi_30_semi_long_0',
        seed=1,
        device='cpu',
        num_threads=3,
        verbose=False,
    )