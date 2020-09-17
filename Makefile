run_simulator_main:
	python -m scripts.ssc.run_simulator_main -c $(config_grid) -m $(model)

run_simulator_main_parallel:
	python -m scripts.ssc.run_simulator_main_parallel -c $(config_grid) -m $(model) -n $(n_jobs)

test_topoae:
	python -m scripts.ssc.run_simulator_main -c 'swissroll.swissroll_testing' -m 'topoae'

test_euler_topoae:
	python -m scripts.ssc.run_simulator_main -c 'swissroll.euler_swissroll_testing' -m 'topoae'

test_euler_topoae_parallel:
	python -m scripts.ssc.run_simulator_main_parallel -c 'swissroll.euler_swissroll_testing_parallel' -m 'topoae' -n 2

test_euler_topoae_ext:
	python -m scripts.ssc.run_simulator_main -c 'swissroll.swissroll_testing_euler' -m 'topoae_ext'

test_euler_topoae_ext_parallel:
	python -m scripts.ssc.run_simulator_main_parallel -c 'swissroll.swissroll_testing_euler_parallel' -m 'topoae_ext' -n 2

test_euler_topoae:
	bsub -W 0:10 -R "rusage[mem=1536]" -oo /cluster/home/schsimo/MT/output/test 'make test_euler_topoae'
	bsub -n 2 -W 0:10 -R "rusage[mem=1536]" -oo /cluster/home/schsimo/MT/output/test 'make test_euler_topoae_parallel'

test_euler_topoae_ext:
	bsub -W 0:10 -R "rusage[mem=1536]" -oo /cluster/home/schsimo/MT/output/test 'make test_euler_topoae_ext'
	bsub -n 2 -W 0:10 -R "rusage[mem=1536]" -oo /cluster/home/schsimo/MT/output/test 'make test_euler_topoae_ext_parallel'



build:
	docker build -t vae-tda	.

test_docker: build
	docker run --runtime=nvidia python scripts/ssc/COREL/test_simulator.py

# deprecated but can still be used....
run_WCTopoAE_main:
	echo $(config_grid)
	python -m scripts.ssc.TopoAE_ext.main_topoae_ext -c $(config_grid)


run_WCTopoAE_parallel:
	echo $(config_grid)
	python -m scripts.ssc.TopoAE_ext.main_topoae_ext_parallel -c $(config_grid)

