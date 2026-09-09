PYTHON ?= python3
APP_FILES = \
	src/gedicorrect/__init__.py \
	src/gedicorrect/__main__.py \
	src/gedicorrect/cli.py \
	src/gedicorrect/config.py \
	src/gedicorrect/jobs.py \
	src/gedicorrect/runner.py \
	src/gedicorrect/system.py \
	src/gedicorrect/ui.py \
	src/gedicorrect/ui_launcher.py

.PHONY: test lint build clean docker

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -v

lint:
	$(PYTHON) -m ruff check $(APP_FILES) src/gedicorrect/utilities tests gedi_correct.py

build:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

clean:
	rm -rf build dist src/GEDICorrect.egg-info

docker:
	docker build --tag gedicorrect:1.0.0 .
