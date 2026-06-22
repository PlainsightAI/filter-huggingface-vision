# ---------------------------------
# Repo-specific variables
# ---------------------------------

VERSION ?= $(shell cat VERSION)
export VERSION

# Define these variables for consistency in the repo
REPO_NAME ?= filter-huggingface-vision
REPO_NAME_SNAKECASE ?= filter_huggingface_vision
REPO_NAME_PASCALCASE ?= FilterHuggingfaceVision

# Unique pipeline configuration for this repo
PIPELINE := \
	- VideoIn \
		--sources 'file://filter_example_video.mp4!loop' \
	- $(REPO_NAME_SNAKECASE).filter.$(REPO_NAME_PASCALCASE) \
		--mq_log pretty \
	- Webvis

IMAGE ?= plainsightai/openfilter-huggingface-vision
CONTAINER_EXEC := docker

check-tag = !(git rev-parse -q --verify "refs/tags/${VERSION}" > /dev/null 2>&1) || \
	(echo "the version: ${VERSION} has been released already" && exit 1)

# ---------------------------------
# Repo-specific targets
# ---------------------------------

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:  ## Install package with dev dependencies
	pip install -e .[dev] \
		--index-url https://python.openfilter.io/simple \
		--extra-index-url https://pypi.org/simple

.PHONY: install-gpu
install-gpu:  ## Install with CUDA 12.8 PyTorch for GPU hosts (driver <= 12.8, e.g. A10); avoids an incompatible CUDA-13 wheel
	pip install -e .[dev] \
		-c constraints-cuda.txt \
		--index-url https://python.openfilter.io/simple \
		--extra-index-url https://download.pytorch.org/whl/cu128 \
		--extra-index-url https://pypi.org/simple

.PHONY: run
run:  ## Run locally with supporting Filters in other processes
	openfilter run ${PIPELINE}

.PHONY: test
test:  ## Run unit tests
	@mkdir -p results
	pytest -vv -s tests/ --junitxml=results/pytest-results.xml

.PHONY: test-coverage
test-coverage:  ## Run unit tests and generate coverage report
	@mkdir -p Reports
	@pytest -vv --cov=tests --junitxml=Reports/coverage.xml --cov-report=json:Reports/coverage.json -s tests/
	@jq -r '["File Name", "Statements", "Missing", "Coverage%"], (.files | to_entries[] | [.key, .value.summary.num_statements, .value.summary.missing_lines, .value.summary.percent_covered_display]) | @csv'  Reports/coverage.json >  Reports/coverage_report.csv
	@jq -r '["TOTAL", (.totals.num_statements // 0), (.totals.missing_lines // 0), (.totals.percent_covered_display // "0")] | @csv'  Reports/coverage.json >>  Reports/coverage_report.csv

.PHONY: build-wheel
build-wheel:  ## Build python wheel
	python -m pip install setuptools build wheel twine setuptools-scm --index-url https://pypi.org/simple
	python -m build --wheel

.PHONY: clean
clean:  ## Delete all generated files and directories
	sudo rm -rf build/ cache/ dist/ $(REPO_NAME_SNAKECASE).egg-info/ telemetry/
	find . -name __pycache__ -type d -exec rm -rf {} +

.PHONY: lint
lint:  ## Run code linting
	@echo "Running flake8..."
	flake8 filter_huggingface_vision/ tests/

.PHONY: format
format:  ## Format code using black and isort
	@echo "Formatting code with black..."
	black filter_huggingface_vision/ tests/
	@echo "Sorting imports with isort..."
	isort filter_huggingface_vision/ tests/

.PHONY: format.check
format.check:  ## Check code formatting
	@echo "Checking code formatting with black..."
	black --check filter_huggingface_vision/ tests/
	@echo "Checking imports with isort..."
	isort --check-only filter_huggingface_vision/ tests/

.PHONY: check-version
check-version:  ## Check if VERSION has already been released/tagged
	@$(check-tag)

.PHONY: publish
publish:  ## Tag with VERSION and git push
	@$(check-tag)
	git tag ${VERSION}
	git push origin ${VERSION}

.PHONY: publish-wheel
publish-wheel: build-wheel  ## Publish python wheel
	TWINE_USERNAME=${PYPI_USERNAME} TWINE_PASSWORD=${PYPI_API_KEY} twine upload dist/*

.PHONY: build-image
build-image:  ## Build docker image
	${CONTAINER_EXEC} build \
		-t ${IMAGE}:${VERSION} \
		--platform linux/amd64,linux/arm64 \
		.

.PHONY: publish-image
publish-image:  ## Publish docker image
	${CONTAINER_EXEC} push ${IMAGE}:${VERSION}

.PHONY: run-image
run-image:  ## Run image in docker container
	${CONTAINER_EXEC} compose up
