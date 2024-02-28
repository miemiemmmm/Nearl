install: 
		echo "Installing the package";


test: 
		cd logs;
		ls -l;
		# logit "Checking the installed package and do some tests";
		python -c "import nearl; print(nearl.__version__)";
		python -c "from nearl import tests; ";

