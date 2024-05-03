create_env:
	sudo conda env create -f env/environment.yaml
	conda info --envs

delete_env:
	sudo conda env remove -n memreservoir

update:
	pip install -r env/requirements.txt