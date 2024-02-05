""" Setup file for synthesizer.

Most the of the build is defined in pyproject.toml but C extensions are not
supported in pyproject.toml yet. To enable the compilation of the C extensions
we use the legacy setup.py. This is ONLY used for the C extensions.
"""
import tempfile

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError

import numpy as np


def has_flags(compiler, flags):
    """
    A function to check whether the C compiler allows for a flag to be passed.

    This is tested by compiling a small temporary test program.

    Args:
        compiler
            The loaded C compiler.
        flags (list)
            A list of compiler flags to test the compiler with.

    Returns
        bool
            Success/Failure
    """

    # Attempt to compile a temporary C file
    with tempfile.NamedTemporaryFile("w", suffix=".c") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=flags)
        except CompileError:
            return False
    return True


class BuildExt(build_ext):
    """
    A class for building extensions with a specific set of accepted compiler
    flags.

    NOTE: Windows is currently not explictly supported.
    """

    # Never check these; they're always added.
    # Note that we don't support MSVC here.
    compile_flags = {
        "unix": [
            "-std=c++17",
            "-w",
            "-O3",
            "-ffast-math",
            "-I{:s}".format(np.get_include()),
        ]
    }

    def build_extensions(self):
        """
        A method to set up the build extensions with the correct compiler flags.
        """

        # Get local useful variables
        ct = self.compiler.compiler_type
        print(ct)

        # Set up the flags and links for compilation
        # opts = self.compile_flags.get(ct, [])
        opts = []
        opts.extend(
            [
                "-std=c++17",
                "-w",
                "-O3",
                "-ffast-math",
                "-I{:s}".format(np.get_include()),
            ]
        )
        links = []
        print(opts)

        # Uncomment below if we use openMP and find a nice way to demonstrate
        # the user how to install it.
        # Will - NOTE: this broke for me with a complaint about -lgomp

        # Check for the presence of -fopenmp; if it's there we're good to go!
        if has_flags(self.compiler, ["-pthread"]):
            # Generic case, this is what GCC accepts
            opts += ["-pthread"]
            links += ["-lpthread"]

        # Apply the flags and links
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = links
            print(ext.extra_compile_args)
            print(ext.extra_link_args)

        # Build the extensions
        build_ext.build_extensions(self)


# Define the extension source files
src_files = {
    "synthesizer.extensions.integrated_spectra": "src/synthesizer/extensions/integrated_spectra.cpp",
    "synthesizer.extensions.particle_spectra": "src/synthesizer/extensions/particle_spectra.cpp",
    "synthesizer.extensions.sfzh": "src/synthesizer/extensions/sfzh.cpp",
    "synthesizer.extensions.los": "src/synthesizer/extensions/los.cpp",
    "synthesizer.imaging.extensions.spectral_cube": "src/synthesizer/imaging/extensions/spectral_cube.cpp",
    "synthesizer.imaging.extensions.image": "src/synthesizer/imaging/extensions/image.cpp",
}

# Create the extension objects
extensions = [
    Extension(
        path,
        sources=[source],
        include_dirs=[np.get_include()],
        py_limited_api=True,
        extra_compile_args=[
            "-std=c++17",
            "-w",
            "-O3",
            "-ffast-math",
            "-I{:s}".format(np.get_include()),
            "-pthread",
        ],
        extra_link_args=["-lpthread"],
    )
    for path, source in src_files.items()
]

# Finally, call the setup
setup(
    ext_modules=extensions,
)
