OUTBASE := /vae-tda/output


prepare_euler:
	bash scripts/ssc/TopoAE/euler_scripts/prepare_euler


# TopoAE
test_TopoAE_euler:
	python -m scripts.ssc.TopoAE.euler_scripts.test_topoae_euler

test_TopoAE_euler_parallel:
	python -m scripts.ssc.TopoAE.euler_scripts.test_topoae_euler_parallel

run_TopoAE_euler_spheres_1:
	python -m scripts.ssc.TopoAE.euler_scripts.run_topoae_euler_spheres_1

run_TopoAE_euler_spheres_2:
	python -m scripts.ssc.TopoAE.euler_scripts.run_topoae_euler_spheres_2

run_TopoAE_euler_swissroll_1:
	python -m scripts.ssc.TopoAE.euler_scripts.run_topoae_euler_swissroll_1

run_TopoAE_euler_swissroll_2:
	python -m scripts.ssc.TopoAE.euler_scripts.run_topoae_euler_swissroll_2

run_TopoAE_euler_spheres_parallel:
	python -m scripts.ssc.TopoAE.euler_scripts.run_topoae_euler_parallel_spheres

run_TopoAE_euler_swissroll:
	python -m scripts.ssc.TopoAE.euler_scripts.run_topoae_euler_parallel_swissroll


# WitnessComplexTopoAE
test_WCTopoAE_euler_swissroll:
	python -m scripts.ssc.TopoAE_ext.euler_scripts.test_wctopoae_euler

#Var = 'scripts.ssc.TopoAE_ext.config_libraries.swissroll.swissroll_testing'
run_WCTopoAE_main:
	echo $(config_grid)
	python -m scripts.ssc.TopoAE_ext.main_topoae_ext -c $(config_grid)


run_WCTopoAE_parallel:
	echo $(config_grid)
	python -m scripts.ssc.TopoAE_ext.main_topoae_ext_parallel -c $(config_grid)


build:
	docker build -t vae-tda	.

test_docker: build
	docker run --runtime=nvidia python scripts/ssc/COREL/test_simulator.py

#-m scripts.ssc.COREL.test_simulator

