.PHONY: help clean clean-pyc clean-test build html

help:
	@echo "Sphinx documentation targets:"
	@echo "  make html       - Build HTML documentation"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make clean-pyc  - Remove Python file artifacts"
	@echo "  make clean-test - Remove test artifacts"

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -delete

clean-pyc:
	find . -type f -name '*.py[cod]' -delete
	find . -type f -name '*~' -delete
	find . -type d -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

html:
	sphinx-build -b html -d _build/doctrees . _build/html
	@echo "Build finished. The HTML pages are in _build/html."
