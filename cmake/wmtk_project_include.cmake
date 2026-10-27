# Included at the end of wildmeshing-toolkit's project() call (CMAKE_PROJECT_WildMeshingToolkit_INCLUDE).

# Defining wmtk::data here makes its wmtk_data recipe return early instead of cloning the ~200 MB
# test-data repository, which only its unit tests read (through WMTK_DATA_DIR).
add_library(wmtk_data INTERFACE)
add_library(wmtk::data ALIAS wmtk_data)

# Some dependencies declare a CMake version older than CMake 4 accepts. The cache variable passed to this
# build doesn't reach the CMake runs they start themselves (fast-envelope's geogram download on Windows),
# but the environment variable does.
set(ENV{CMAKE_POLICY_VERSION_MINIMUM} 3.5)

# lagrange's bundled mapbox/earcut.h uses uint32_t without including <cstdint>, which GCC 13 and later
# no longer include transitively.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    string(APPEND CMAKE_CXX_FLAGS " -include cstdint")
endif()
