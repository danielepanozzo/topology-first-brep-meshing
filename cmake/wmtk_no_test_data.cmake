# Included at the end of wildmeshing-toolkit's project() call. Defining wmtk::data here makes its
# wmtk_data recipe return early instead of cloning the ~200 MB test-data repository, which only its
# unit tests read (through WMTK_DATA_DIR).
add_library(wmtk_data INTERFACE)
add_library(wmtk::data ALIAS wmtk_data)
