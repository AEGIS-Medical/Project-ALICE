# Project ALICE -- developer Makefile.
#
# Requires GNU Make. On Windows install via Chocolatey (`choco install make`)
# or use Git Bash + WSL. The recipe shells assume POSIX sh syntax.

.PHONY: install test-compress proto live

PYTHON ?= python
MODE   ?= raw

install:
	$(PYTHON) -m pip install -e ".[dev]"

# Run the compression pipeline against a single video.
#
# Usage:
#   make test-compress VIDEO=demo_data/honest/demo_file_1/clip.mp4
#   make test-compress VIDEO=path/to/video.mp4 MODE=edge_full
#
# MODE defaults to "raw"; valid values: raw | roi | edge_full | edge_minimal.
test-compress:
	@if [ -z "$(VIDEO)" ]; then \
		echo "Usage: make test-compress VIDEO=path/to/video.mp4 [MODE=raw|roi|edge_full|edge_minimal]"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make test-compress VIDEO=demo_data/honest/demo_file_1/clip.mp4"; \
		echo "  make test-compress VIDEO=path/to/video.mp4 MODE=edge_full"; \
		exit 2; \
	fi
	$(PYTHON) scripts/test_compression.py "$(VIDEO)" --mode $(MODE)

# Regenerate protobuf codegen after editing proto/landmarks.proto.
# The generated file is COMMITTED; normal builds never need this.
proto:
	$(PYTHON) -m grpc_tools.protoc -Iproto --python_out=backend/shared/proto_gen proto/landmarks.proto

# Run the live service (dev posture: localhost, no auth).
live:
	$(PYTHON) scripts/run_live_service.py
