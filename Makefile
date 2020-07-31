OUTBASE := /vae-tda/output


prepare_euler:
	bash scripts/ssc/TopoAE/euler_scripts/prepare_euler



run_TopoAE_euler:
	bash pip --no-cache-dir install -r requirements.txt --user
	bash pip install dataclasses --user
	bash pip3 install torch torchvision --user
	bash pip install numpy --upgrade --user
	bash pip install scikit-learn --upgrade --user
	bash pip install umap --user
	bash pip install umap-learn --user
	python -m scripts.ssc.TopoAE.euler_scripts.run_topoae_euler


build:
	docker build -t vae-tda	.

test_docker: build
	docker run --runtime=nvidia python scripts/ssc/COREL/test_simulator.py

#-m scripts.ssc.COREL.test_simulator