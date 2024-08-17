# Author: Yang Zhang 
# Date: 2024-07-13
# Description: Makefile for the project


install: clean
	@version=$$(grep "version" pyproject.toml | sed -n 's/version = "\([0-9.]*\)"/\1/p'); \
	echo "Installing the package; Version $${version}"; \
	python -m build; \
	pip install -v ./dist/nearl-$${version}-py3-none-any.whl --force-reinstall; 
	$(MAKE) clean


# print:
# 	echo "Version: $$version";

install_dependencies: 
	echo "install_dependencies; "


# TODO: Add the installation later
install_mamba: 
	echo "micromamba; " 


clean: 
	rm -rf dist/ build/ nearl.egg-info/ .pytest_cache/ 


dotest: 
	python3 -c "import nearl; print(nearl.__version__)"
	pytest nearl/tests


compilepdf: 
	cd docs && docker run --rm --volume ./:/data --env JOURNAL=joss openjournals/inara


compilehtml: 
	cd docs && make html
