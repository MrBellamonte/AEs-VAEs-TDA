OUTBASE := /vae-tda/output


prepare_euler:
	bash scripts/ssc/TopoAE/euler_scripts/prepare_euler


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
	python -m scripts.ssc.TopoAE.euler_scripts.run_topoae_euler_spheres_parallel

run_TopoAE_euler_swissroll:
	python -m scripts.ssc.TopoAE.euler_scripts.run_topoae_euler_parallel_swissroll


build:
	docker build -t vae-tda	.

test_docker: build
	docker run --runtime=nvidia python scripts/ssc/COREL/test_simulator.py

#-m scripts.ssc.COREL.test_simulator

