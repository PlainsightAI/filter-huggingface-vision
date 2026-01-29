# Common variables
PYPI_REPO ?= https://us-west1-python.pkg.dev/plainsightai-prod/python/simple/
VERSION ?= $(shell cat VERSION)
RESOURCE_BUNDLE_VERSION ?= $(shell cat RESOURCE_BUNDLE_VERSION)
CONTAINER_EXEC := docker
GOOGLE_APPLICATION_CREDENTIALS ?= $(HOME)/.config/gcloud/application_default_credentials.json

export IMAGE
export MODEL_IMAGE
export VERSION
export RESOURCE_BUNDLE_VERSION

check-tag = !(git rev-parse -q --verify "refs/tags/v${VERSION}" > /dev/null 2>&1) || \
	(echo "the version: ${VERSION} has been released already" && exit 1)


.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


.PHONY: check-version
check-version:  ## Check if VERSION has already been released/tagged
	@$(check-tag)


.PHONY: publish
publish:  ## Tag with VERSION and git push
	@$(check-tag)
	git tag v${VERSION}
	git push origin v${VERSION}


.PHONY: build-wheel
build-wheel:  ## Build python wheel
	python -m pip install setuptools build wheel twine setuptools-scm --index-url https://pypi.org/simple
	python -m build --wheel


.PHONY: publish-wheel
publish-wheel:  ## Publish python wheel
	TWINE_USERNAME=${PYPI_USERNAME} TWINE_PASSWORD=${PYPI_API_KEY} twine upload --repository-url ${PYPI_REPO} dist/*


.PHONY: build-image
build-image:  ## Build docker image
	${CONTAINER_EXEC} build \
		-t ${IMAGE}:${VERSION} \
		--build-arg RESOURCE_BUNDLE_VERSION=${RESOURCE_BUNDLE_VERSION} \
		--secret id=google_credentials,src=${GOOGLE_APPLICATION_CREDENTIALS} \
		.

.PHONY: publish-image
publish-image:  ## Publish docker image
	${CONTAINER_EXEC} push ${IMAGE}:${VERSION}

.PHONY: build-model-image
build-model-image:  ## Build the model image
	docker build -t $(MODEL_IMAGE):$(RESOURCE_BUNDLE_VERSION) -t $(MODEL_IMAGE):latest -f Dockerfile.model .


.PHONY: check-version-exists
check-version-exists:  ## Check if the version already exists in GAR
	@RAW_OUTPUT=$$(gcloud artifacts docker tags list $(MODEL_IMAGE) \
		--format="table(tag)" 2>&1); \
	MATCH=$$(echo "$$RAW_OUTPUT" | tail -n +2 | grep -Fx "$(VERSION)" || true); \
	if [ -n "$$MATCH" ]; then \
		echo "❌ Error: Version '$(VERSION)' already exists in GAR."; \
		exit 1; \
	else \
		echo "✅ Version '$(VERSION)' does not exist in GAR."; \
	fi

.PHONY: publish-model-image
publish-model-image: check-version-exists ## Publish the model image
	docker push $(MODEL_IMAGE):$(RESOURCE_BUNDLE_VERSION)
	docker push $(MODEL_IMAGE):latest


.PHONY: run-image
run-image:  ## Run image in docker container
	${CONTAINER_EXEC} compose up


.PHONY: run
run:  ## Run locally with supporting Filters in other processes
	openfilter run ${PIPELINE}


.PHONY: test
test:  ## Run unit tests
	pytest -vv -s tests/ --junitxml=results/pytest-results.xml


.PHONY: test-coverage
test-coverage:  ## Run unit tests and generate coverage report
	@mkdir -p Reports
	@pytest -vv --cov=tests --junitxml=Reports/coverage.xml --cov-report=json:Reports/coverage.json -s tests/
	@jq -r '["File Name", "Statements", "Missing", "Coverage%"], (.files | to_entries[] | [.key, .value.summary.num_statements, .value.summary.missing_lines, .value.summary.percent_covered_display]) | @csv'  Reports/coverage.json >  Reports/coverage_report.csv
	@jq -r '["TOTAL", (.totals.num_statements // 0), (.totals.missing_lines // 0), (.totals.percent_covered_display // "0")] | @csv'  Reports/coverage.json >>  Reports/coverage_report.csv


.PHONY: clean
clean:  ## Delete all generated files and directories
	sudo rm -rf build/ cache/ dist/ $(REPO_NAME_SNAKECASE).egg-info/ telemetry/
	find . -name __pycache__ -type d -exec rm -rf {} +

