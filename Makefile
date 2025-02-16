# Author: Yang Zhang 
# Date: 2024-07-13
# Description: Makefile for the project

MAMBA_ROOT_PREFIX ?= /tmp/micromamba

install: clean
	@version=$$(grep "version" pyproject.toml | sed -n 's/version = "\([0-9.]*\)"/\1/p'); \
	echo "Installing the package; Version $${version}"; \
	python -m build; \
	pip install -v ./dist/nearl-$${version}-py3-none-any.whl --force-reinstall; 
	$(MAKE) clean


reinstall: 
	pip install --force-reinstall ./ 


install_mamba: 
	echo "Installing mamba into MAMBA_ROOT_PREFIX -> $(MAMBA_ROOT_PREFIX); "
	curl -s https://gist.githubusercontent.com/miemiemmmm/40d2e2b49e82d682ef5a7b2aa94a243f/raw/b9a3e3c916cbee42b2cfedcda69d2db916e637c0/install_micromamba.sh | bash -s -- $(MAMBA_ROOT_PREFIX)


# TODO: Add the installation later 
install_dependencies: 
	$(MAMBA_ROOT_PREFIX)/bin/micromamba install -f requirements.yml -y 


clean: 
	rm -rf dist/ build/ nearl.egg-info/ .pytest_cache/ 


dotest: 
	cd pytests && python3 -m pytest -v --import-mode=importlib 
	cd pytests && python3 -m pytest -v --benchmark-only --import-mode=importlib --benchmark-min-rounds=100 


document: 
	cd docs && make html

