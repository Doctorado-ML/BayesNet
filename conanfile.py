import os, re, pathlib
from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps
from conan.tools.files import copy


class BayesNetConan(ConanFile):
    name = "bayesnet"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "enable_testing": [True, False],
        "enable_coverage": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "enable_testing": False,
        "enable_coverage": False,
    }

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = (
        "CMakeLists.txt",
        "bayesnet/*",
        "config/*",
        "cmake/*",
        "docs/*",
        "tests/*",
        "bayesnetConfig.cmake.in",
    )

    def set_version(self) -> None:
        cmake = pathlib.Path(self.recipe_folder) / "CMakeLists.txt"
        text = cmake.read_text(encoding="utf-8")

        # Accept either: project(foo VERSION 1.2.3)  or  set(foo_VERSION 1.2.3)
        match = re.search(
            r"""project\s*\([^\)]*VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)""",
            text,
            re.IGNORECASE | re.VERBOSE,
        )
        if match:
            self.version = match.group(1)
        else:
            raise Exception("Version not found in CMakeLists.txt")
        self.version = match.group(1)

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def requirements(self):
        # Core dependencies
        self.requires("libtorch/2.7.1")
        self.requires("nlohmann_json/3.11.3")
        self.requires("folding/1.1.2")  # Custom package
        self.requires("fimdlp/2.1.1")  # Custom package

    def build_requirements(self):
        self.build_requires("cmake/[>=3.27]")
        self.test_requires("arff-files/1.2.1")  # Custom package
        self.test_requires("catch2/3.8.1")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.variables["ENABLE_TESTING"] = self.options.enable_testing
        tc.variables["CODE_COVERAGE"] = self.options.enable_coverage
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

        if self.options.enable_testing:
            # Run tests only if we're building with testing enabled
            self.run("ctest --output-on-failure", cwd=self.build_folder)

    def package(self):
        copy(
            self,
            "LICENSE",
            src=self.source_folder,
            dst=os.path.join(self.package_folder, "licenses"),
        )
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["bayesnet"]
        self.cpp_info.includedirs = ["include"]
        self.cpp_info.set_property("cmake_find_mode", "both")
        self.cpp_info.set_property("cmake_target_name", "bayesnet::bayesnet")

        # Add compiler flags that might be needed
        if self.settings.os == "Linux":
            self.cpp_info.system_libs = ["pthread"]
