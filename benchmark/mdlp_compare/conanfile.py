from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps


class MdlpBenchConan(ConanFile):
    """Dependencies of the mdlp comparison benchmark.

    The fimdlp version is an option so that the very same sources can be built
    twice, once per version under test:

        conan install . -o mdlp_version=2.1.3 -of build_2.1.3 -s build_type=Release
        conan install . -o mdlp_version=3.0.0 -of build_3.0.0 -s build_type=Release
    """

    name = "mdlp_bench"
    settings = "os", "compiler", "build_type", "arch"
    options = {"mdlp_version": ["2.1.3", "3.0.0"]}
    default_options = {"mdlp_version": "2.1.3"}

    def requirements(self):
        self.requires("libtorch/2.7.1")
        self.requires("nlohmann_json/3.11.3")
        self.requires("folding/2.0.0")
        self.requires("arff-files/1.2.1")
        self.requires("fimdlp/{}".format(self.options.mdlp_version))

    def generate(self):
        CMakeDeps(self).generate()
        CMakeToolchain(self).generate()
