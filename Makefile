OUTBASE := /vae-tda/output


prepare_euler:
	bash scripts/ssc/TopoAE/euler_scripts/prepare_euler


test_TopoAE_euler:
	python -m scripts.ssc.TopoAE.euler_scripts.test_topoae_euler

run_TopoAE_euler_spheres:
	python -m scripts.ssc.TopoAE.euler_scripts.run_topoae_euler_spheres

run_TopoAE_euler_swissroll:
	python -m scripts.ssc.TopoAE.euler_scripts.run_topoae_euler_swissroll


build:
	docker build -t vae-tda	.

test_docker: build
	docker run --runtime=nvidia python scripts/ssc/COREL/test_simulator.py

#-m scripts.ssc.COREL.test_simulator

