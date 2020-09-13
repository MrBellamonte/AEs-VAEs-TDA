run_simulator_main:
	python -m scripts.ssc.run_simulator_main -c $(config_grid) -m $(model)

run_simulator_main_parallel:
	python -m scripts.ssc.run_simulator_main_parallel -c $(config_grid) -m $(model) -n $(n_jobs)


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

