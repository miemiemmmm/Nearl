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

document_dependencies: 
	pip install sphinx-copybutton sphinx-togglebutton myst-parser sphinx-design numpydoc sphinx_book_theme
