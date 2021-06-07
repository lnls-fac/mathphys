## ------------------------
## mathphys - Makefile help
## ------------------------

PACKAGE = $(shell basename $(shell pwd))

install: clean ## Install packge using the local repository
	sudo ./setup.py install --single-version-externally-managed --compile --force --record /dev/null

develop: clean ## Install in editable mode (i.e. setuptools "develop mode")
	sudo ./setup.py develop

clean: ## Clean repository via "git clean -fdX"
	git clean -fdX

help:  ## Show this help.
	@grep '##' Makefile| sed -e '/@/d' | sed -r 's,(.*?:).*##(.*),\1\2,g'

dist: clean ## Build setuptools dist
	python setup.py sdist bdist_wheel

distupload: ## Upload package dist to PyPi
	python -m twine upload --verbose dist/*

distinstall: ## Install package from PyPi
	python -m pip install $(PACKAGE)==$(shell cat "VERSION")

disttestupload: ##  Upload package dist to Test PyPi
	python -m twine upload --verbose --repository testpypi dist/*

disttestinstall: ##  Install package from Test PyPi
	python -m pip install --index-url https://test.pypi.org/simple/ --no-deps $(PACKAGE)==$(shell cat "VERSION")

disttest: dist disttestupload disttestinstall test ## Build the package, upload to Test PyPi, install from PyPi and run tests

