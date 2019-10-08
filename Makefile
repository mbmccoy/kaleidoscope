IMAGE_NAME="mbmccoy/kaleidoscope"

Pipfile.lock: Pipfile
	pipenv install --dev

test:
	docker build \
		-t ${IMAGE_NAME} \
		-f Dockerfile .

shell: image
	@docker run \
		--interactive \
		--tty \
		${IMAGE_NAME} \
		/bin/bash

.PHONY: image
