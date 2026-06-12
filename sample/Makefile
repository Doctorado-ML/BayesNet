SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help init build run run-xspode clean check-conan

# ---------------------------------------------------------------------------
# Configurable variables (override on the command line: make run model=AODE)
# ---------------------------------------------------------------------------
build_type ?= Release
build_dir  ?= build
fname      ?= ../tests/data/iris.arff
model      ?= TANLd
sp         ?= 0

CONAN_REMOTE_NAME ?= Cimmeria
CONAN_REMOTE_URL  ?= https://conan.rmontanana.es/artifactory/api/conan/Cimmeria

CPUS := $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || nproc --all 2>/dev/null || sysctl -n hw.ncpu)
JOBS := $(shell n=$(CPUS); [ $${n} -gt 4 ] && echo $$((n-2)) || echo 1)

GREEN  = \033[0;32m
YELLOW = \033[1;33m
RED    = \033[0;31m
NC     = \033[0m

help: ## Show this help
	@echo ""
	@printf "$(GREEN)BayesNet Sample$(NC) - Available targets:\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-14s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@printf "$(GREEN)Variables$(NC) (override with VAR=value):\n"
	@printf "  $(YELLOW)build_type$(NC)  Release | Debug          (current: $(build_type))\n"
	@printf "  $(YELLOW)build_dir$(NC)   build directory          (current: $(build_dir))\n"
	@printf "  $(YELLOW)fname$(NC)       ARFF dataset path        (current: $(fname))\n"
	@printf "  $(YELLOW)model$(NC)       classifier name          (current: $(model))\n"
	@printf "  $(YELLOW)sp$(NC)          XSpode super parent idx  (current: $(sp))\n"
	@echo ""
	@printf "$(GREEN)Examples:$(NC)\n"
	@printf "  make all                                # init + build + run with defaults\n"
	@printf "  make run model=BoostAODE                # use a different model\n"
	@printf "  make run fname=../tests/data/glass.arff model=AODELd\n"
	@printf "  make run-xspode fname=../tests/data/iris.arff sp=2\n"
	@echo ""

check-conan: ## Verify Conan is installed and the Cimmeria remote is configured
	@command -v conan >/dev/null 2>&1 || { \
		printf "$(RED)ERROR:$(NC) Conan is not installed.\n  Install it with: pip install conan\n"; exit 1; }
	@if ! conan remote list 2>/dev/null | grep -q "^$(CONAN_REMOTE_NAME):"; then \
		printf "$(YELLOW)Adding Conan remote '$(CONAN_REMOTE_NAME)'$(NC)\n"; \
		conan remote add $(CONAN_REMOTE_NAME) $(CONAN_REMOTE_URL); \
	fi
	@if ! conan profile list 2>/dev/null | grep -q "^default$$"; then \
		printf "$(YELLOW)Detecting default Conan profile$(NC)\n"; \
		conan profile detect --force; \
	fi

init: check-conan ## Install dependencies via Conan
	@printf "$(GREEN)>>> Installing dependencies ($(build_type))$(NC)\n"
	@conan install . --output-folder=$(build_dir) --build=missing -s build_type=$(build_type)

build: init ## Configure CMake and build both sample binaries
	@printf "$(GREEN)>>> Configuring CMake ($(build_type))$(NC)\n"
	@cmake -B $(build_dir) -S . \
		-DCMAKE_BUILD_TYPE=$(build_type) \
		-DCMAKE_TOOLCHAIN_FILE=$(build_dir)/conan_toolchain.cmake
	@printf "$(GREEN)>>> Building with $(JOBS) parallel jobs$(NC)\n"
	@cmake --build $(build_dir) --parallel $(JOBS)

run: build ## Build and run bayesnet_sample (use fname=... model=...)
	$(eval ABS_FNAME := $(shell python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" $(fname)))
	@printf "$(GREEN)>>> Running bayesnet_sample $(ABS_FNAME) $(model)$(NC)\n"
	@./$(build_dir)/bayesnet_sample $(ABS_FNAME) $(model)

run-xspode: build ## Build and run bayesnet_sample_xspode (use fname=... sp=...)
	$(eval ABS_FNAME := $(shell python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" $(fname)))
	@printf "$(GREEN)>>> Running bayesnet_sample_xspode $(ABS_FNAME) $(sp)$(NC)\n"
	@./$(build_dir)/bayesnet_sample_xspode $(ABS_FNAME) $(sp)

all: run ## Alias for: init + build + run

clean: ## Remove the build directory
	@printf "$(GREEN)>>> Cleaning $(build_dir)$(NC)\n"
	@rm -rf $(build_dir)
