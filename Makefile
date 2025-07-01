SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: viewcoverage coverage setup help install uninstall diagrams buildr buildd test clean updatebadge doc doc-install init clean-test conan-debug conan-release conan-create conan-upload conan-clean conan-sample

f_release = build_Release
f_debug = build_Debug
f_diagrams = diagrams
app_targets = bayesnet
test_targets = TestBayesNet
clang-uml = clang-uml
plantuml = plantuml
lcov = lcov
genhtml = genhtml
dot = dot
docsrcdir = docs/manual
mansrcdir = docs/man3
mandestdir = /usr/local/share/man
sed_command_link = 's/e">LCOV -/e"><a href="https:\/\/rmontanana.github.io\/bayesnet">Back to manual<\/a> LCOV -/g'
sed_command_diagram = 's/Diagram"/Diagram" width="100%" height="100%" /g'

define ClearTests
	@for t in $(test_targets); do \
		if [ -f $(f_debug)/tests/$$t ]; then \
			echo ">>> Cleaning $$t..." ; \
			rm -f $(f_debug)/tests/$$t ; \
		fi ; \
	done
	@nfiles="$(find . -name "*.gcda" -print0)" ; \
	if test "${nfiles}" != "" ; then \
		find . -name "*.gcda" -print0 | xargs -0 rm 2>/dev/null ;\
	fi ; 
endef

setup: ## Install dependencies for tests and coverage
	@if [ "$(shell uname)" = "Darwin" ]; then \
		brew install gcovr; \
		brew install lcov; \
	fi
	@if [ "$(shell uname)" = "Linux" ]; then \
		pip install gcovr; \
		sudo dnf install lcov;\
	fi
	@echo "* You should install plantuml & graphviz for the diagrams"

clean: ## Clean the project
	@echo ">>> Cleaning the project..."
	@if test -f CMakeCache.txt ; then echo "- Deleting CMakeCache.txt"; rm -f CMakeCache.txt; fimake 
	@for folder in $(f_release) $(f_debug) vpcpkg_installed install_test ; do \
	if test -d "$$folder" ; then \
		echo "- Deleting $$folder folder" ; \
		rm -rf "$$folder"; \
	fi; \
	done
	@$(MAKE) clean-test
	@echo ">>> Done";

# Build targets
# =============

buildd: ## Build the debug targets
	cmake --build $(f_debug) --config Debug -t $(app_targets) --parallel $(CMAKE_BUILD_PARALLEL_LEVEL)

buildr: ## Build the release targets
	cmake --build $(f_release) --config Release -t $(app_targets) --parallel $(CMAKE_BUILD_PARALLEL_LEVEL)


# Install targets
# ===============

uninstall: ## Uninstall library
	@echo ">>> Uninstalling BayesNet...";
	xargs rm < $(f_release)/install_manifest.txt
	@echo ">>> Done";

prefix = "/usr/local"
install: ## Install library
	@echo ">>> Installing BayesNet...";
	@cmake --install $(f_release) --prefix $(prefix)
	@echo ">>> Done";


# Test targets
# ============

clean-test: ## Clean the tests info
	@echo ">>> Cleaning Debug BayesNet tests...";
	$(call ClearTests)
	@echo ">>> Done";

opt = ""
test: ## Run tests (opt="-s") to verbose output the tests, (opt="-c='Test Maximum Spanning Tree'") to run only that section
	@echo ">>> Running BayesNet tests...";
	@$(MAKE) clean-test
	@cmake --build $(f_debug) -t $(test_targets) --parallel $(CMAKE_BUILD_PARALLEL_LEVEL)
	@for t in $(test_targets); do \
		echo ">>> Running $$t...";\
		if [ -f $(f_debug)/tests/$$t ]; then \
			cd $(f_debug)/tests ; \
			./$$t $(opt) ; \
			cd ../.. ; \
		fi ; \
	done
	@echo ">>> Done";

coverage: ## Run tests and generate coverage report (build/index.html)
	@echo ">>> Building tests with coverage..."
	@which $(lcov) || (echo ">>ease install lcov"; exit 1)
	@if [ ! -f $(f_debug)/tests/coverage.info ] ; then $(MAKE) test ; fi
	@echo ">>> Building report..."
	@cd $(f_debug)/tests; \
	$(lcov) --directory CMakeFiles --capture --demangle-cpp --ignore-errors source,source --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --remove coverage.info '/usr/*' --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --remove coverage.info 'lib/*' --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --remove coverage.info 'include/*' --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --remove coverage.info 'libtorch/*' --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --remove coverage.info 'tests/*' --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --remove coverage.info 'bayesnet/utils/loguru.*' --ignore-errors unused --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --remove coverage.info '/opt/miniconda/*' --ignore-errors unused --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --remove coverage.info '*/.conan2/*' --ignore-errors unused --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --summary coverage.info
	@$(MAKE) updatebadge
	@echo ">>> Done";	

viewcoverage: ## View the html coverage report
	@which $(genhtml) >/dev/null || (echo ">>> Please install lcov (genhtml not found)"; exit 1)
	@if [ ! -d $(docsrcdir)/coverage ]; then mkdir -p $(docsrcdir)/coverage; fi
	@if [ ! -f $(f_debug)/tests/coverage.info ]; then \
		echo ">>> No coverage.info file found. Run make coverage first!"; \
		exit 1; \
	fi
	@$(genhtml) $(f_debug)/tests/coverage.info --demangle-cpp --output-directory $(docsrcdir)/coverage --title "BayesNet Coverage Report" -s -k -f --legend >/dev/null 2>&1;
	@xdg-open $(docsrcdir)/coverage/index.html || open $(docsrcdir)/coverage/index.html 2>/dev/null
	@echo ">>> Done";

updatebadge: ## Update the coverage badge in README.md
	@which python || (echo ">>> Please install python"; exit 1)
	@if [ ! -f $(f_debug)/tests/coverage.info ]; then \
		echo ">>> No coverage.info file found. Run make coverage first!"; \
		exit 1; \
	fi
	@echo ">>> Updating coverage badge..."
	@env python update_coverage.py $(f_debug)/tests
	@echo ">>> Done";

# Documentation targets
# =====================

doc: ## Generate documentation
	@echo ">>> Generating documentation..."
	@cmake --build $(f_release) -t doxygen
	@cp -rp diagrams $(docsrcdir)
	@
	@if [ "$(shell uname)" = "Darwin" ]; then \
		sed -i "" $(sed_command_link) $(docsrcdir)/coverage/index.html ; \
		sed -i "" $(sed_command_diagram) $(docsrcdir)/index.html ; \
	else \
		sed -i $(sed_command_link) $(docsrcdir)/coverage/index.html ; \
		sed -i $(sed_command_diagram) $(docsrcdir)/index.html ; \
	fi
	@echo ">>> Done";

diagrams: ## Create an UML class diagram & dependency of the project (diagrams/BayesNet.png)
	@which $(plantuml) || (echo ">>> Please install plantuml"; exit 1)
	@which $(dot) || (echo ">>> Please install graphviz"; exit 1)
	@which $(clang-uml) || (echo ">>> Please install clang-uml"; exit 1)
	@export PLANTUML_LIMIT_SIZE=16384
	@echo ">>> Creating UML class diagram of the project...";
	@$(clang-uml) -p 
	@cd $(f_diagrams); \
	$(plantuml) -tsvg BayesNet.puml
	@echo ">>> Creating dependency graph diagram of the project...";
	$(MAKE) debug
	cd $(f_debug) && cmake .. --graphviz=dependency.dot 
	@$(dot) -Tsvg $(f_debug)/dependency.dot.BayesNet -o $(f_diagrams)/dependency.svg

docdir = ""
doc-install: ## Install documentation
	@echo ">>> Installing documentation..."
	@if [ "$(docdir)" = "" ]; then \
		echo "docdir parameter has to be set when calling doc-install, i.e. docdir=../bayesnet_help"; \
		exit 1; \
	fi
	@if [ ! -d $(docdir) ]; then \
		@$(MAKE) doc; \
	fi
	@cp -rp $(docsrcdir)/* $(docdir)
	@sudo cp -rp $(mansrcdir) $(mandestdir)
	@echo ">>> Done";

# Conan package manager targets
# =============================

debug:             ## Build debug version using Conan
	@echo ">>> Building *Debug* BayesNet with Conan..."
	@rm -rf $(f_debug)            # wipe previous tree
	@conan install . \
	            -s build_type=Debug \
	            --build=missing \
	            -of $(f_debug) \
				--profile=debug
	@cmake -S . -B $(f_debug) \
	       -DCMAKE_BUILD_TYPE=Debug \
	       -DENABLE_TESTING=ON \
	       -DCODE_COVERAGE=ON \
	       -DCMAKE_TOOLCHAIN_FILE=$(f_debug)/build/Debug/generators/conan_toolchain.cmake
	@echo ">>> Done"

release: ## Build release version using Conan
	@echo ">>> Building Release BayesNet with Conan..."
	@conan install . \
	            -s build_type=Release \
	            --build=missing \
	            -of $(f_debug) \
				--profile=release
	@if [ -d ./$(f_release) ]; then rm -rf ./$(f_release); fi
	@mkdir $(f_release)
	@conan install . -s build_type=Release --build=missing -of $(f_release)
	@cmake -S . -B $(f_release) -D CMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$(f_release)/build/Release/generators/conan_toolchain.cmake
	@echo ">>> Done"

conan-create: ## Create Conan package
	@echo ">>> Creating Conan package..."
	@conan create . --build=missing -tf "" --profile=release -tf ""
	@conan create . --build=missing -tf "" --profile=debug
	@echo ">>> Done"

profile ?= release
remote ?= Cimmeria
conan-upload: ## Upload package to Conan remote (profile=release remote=Cimmeria)
	@echo ">>> Uploading to Conan remote $(remote) with profile $(profile)..."
	@conan upload bayesnet/$(grep version conanfile.py | cut -d'"' -f2) -r $(remote) --confirm
	@echo ">>> Done"

conan-clean: ## Clean Conan cache and build folders
	@echo ">>> Cleaning Conan cache and build folders..."
	@conan remove "*" --confirm
	@if test -d "$(f_release)" ; then rm -rf "$(f_release)"; fi
	@if test -d "$(f_debug)" ; then rm -rf "$(f_debug)"; fi
	@echo ">>> Done"

fname = "tests/data/iris.arff"
model = "TANLd"
sample: ## Build sample with Conan
	@echo ">>> Building Sample with Conan...";
	@if [ -d ./sample/build ]; then rm -rf ./sample/build; fi
	@cd sample && conan install . --output-folder=build --build=missing
	@cd sample && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake && \
	cmake --build build -t bayesnet_sample
	sample/build/bayesnet_sample $(fname) $(model)
	@echo ">>> Done";

# Help target
# ===========

help: ## Show help message
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-20s %s\n" "task" "help" ; \
	printf "%-20s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-20s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done