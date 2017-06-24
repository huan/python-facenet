# Makefile for facenet
# https://github.com/zixia/python-facenet
# Author: Huan LI <zixia@zixia.net> git.io/zixia

SOURCE_GLOB=$(wildcard src/**/*.py)

.PHONY: all
all : clean lint

.PHONY: clean
clean:
	echo "TODO: clean what?"

.PHONY: lint
lint: pylint pep8 flake8 mypy

.PHONY: pylint
pylint:
	pylint $(SOURCE_GLOB)

.PHONY: pep8
pep8:
	pep8 $(SOURCE_GLOB)

.PHONY: flake8
flake8:
	flake8 $(SOURCE_GLOB)

.PHONY: mypy
mypy:
	MYPYPATH=stubs/ mypy \
		--disallow-untyped-defs \
		--python-version=3.6 \
		$(SOURCE_GLOB)

.PHONY: test
test: check-version lint
	echo "test"
	# false
	echo "ok"

.PHONY: check-version
check-version:
	./util/check_version.py

