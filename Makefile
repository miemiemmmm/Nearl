install: 
	echo "Installing the package";


dotest: 
	python3 -c "import nearl; print(nearl.__version__)"
	pytest nearl/tests

compilepdf: 
	cd docs && docker run --rm --volume ./:/data --env JOURNAL=joss openjournals/inara

compilehtml: 
	cd docs && make html
# sudo docker run --rm --volume ${PWD}:/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara