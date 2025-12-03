# Maligner

## Python 3 Port Notice

**This is a Python 3 port of the original maligner software.**

The original `maligner` software was written in Python 2 and distributed under the GPL license. This version has been ported to Python 3 for compatibility with modern Python environments.

**Original Source:**
- **GitHub Repository**: [https://github.com/LeeMendelowitz/maligner](https://github.com/LeeMendelowitz/maligner)
- **Publication**: Bioinformatics, Volume 32, Issue 7, April 2016, Pages 1016–1022

For the original source code, documentation, and publications, please refer to the [original maligner GitHub repository](https://github.com/LeeMendelowitz/maligner).

**Modifications made:**
- Ported from Python 2 to Python 3
- Updated syntax and dependencies for Python 3 compatibility
- No functional changes to the core alignment algorithms
- Moved `malignpy` from `build/lib/` to `build/bin/` directory
- Included only the essential files needed to run the software (executable and required dependencies)

**Distribution Note**: This distribution includes only the compiled executable (`build/bin/maligner_dp`) and essential runtime files. The source code for this Python 3 port is available upon request or can be obtained from the original maligner source code repository.

**Original License**: GPL v3.0 (GNU General Public License version 3.0)  
**See**: `LICENSE` file for the full GPL license text

**Source Code Availability**: In accordance with GPL v3.0 requirements, the source code for this Python 3 port is available. The original source code can be obtained from the [original maligner GitHub repository](https://github.com/LeeMendelowitz/maligner). For the Python 3 port source code, please contact the maintainers.

## Platform Compatibility

**Important**: The pre-built `maligner_dp` executable in `build/bin/maligner_dp` is compiled for **Linux** only.

- **Linux**: The included executable should work on Linux systems
- **macOS**: The Linux executable will not run on macOS. However, you only need to rebuild the `maligner_dp` executable for macOS. The `malignpy` package and other Python scripts are platform-independent and can be used as-is (they only require Python 3 compatibility)
- **Windows**: Not supported (would require rebuilding from source)

If you need to use `maligner_dp` on macOS, rebuild only the `maligner_dp` executable from source and replace the one in `build/bin/`. The Python components (`malignpy` and utility scripts) are platform-independent.

## Building

The `maligner_dp` executable is located in `build/bin/maligner_dp`. 

**Note**: The `maligner_dp` executable has been pre-built for Linux. The `malignpy` package (originally located in `build/lib/malignpy`) and related files have been moved to `build/bin/` for convenience. 

**For macOS users**: Only the `maligner_dp` executable needs to be rebuilt for macOS. The `malignpy` package and Python utility scripts are platform-independent and can be used as-is. To rebuild `maligner_dp` for macOS, use the [original maligner repository](https://github.com/LeeMendelowitz/maligner) and apply Python 3 compatibility modifications as needed. Refer to the original repository for build instructions.

## Usage

The `maligner_dp` executable is used by DOM (Dual Optical Mapping) toolkit for alignment operations. See the main DOM README for usage instructions.

