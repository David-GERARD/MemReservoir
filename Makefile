env:
	sudo conda env create -f env/environment.yaml
	conda info --envs
	conda activate memreservoir

update:
	pip install -r env/requirements.txt