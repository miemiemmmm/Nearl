install: 
		echo "Installing the package";


test: 
		cd logs;
		ls -l;
		# logit "Checking the installed package and do some tests";
		python -c "import nearl; print(nearl.__version__)";
		python -c "from nearl import tests; ";

compilepdf: 
		echo "sudo docker run --rm \
		--volume ${PWD}:/data \
		--user $(id -u):$(id -g) \
		--env JOURNAL=joss \
		openjournals/inara";