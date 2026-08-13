# PithosConfig.cmake
# CMake configuration file for importing Pithos C/C++ Native SDK

@PACKAGE_INIT@

find_path(PITHOS_INCLUDE_DIR
    NAMES pithos.h
    PATHS "${PACKAGE_PREFIX_DIR}/include"
)

find_library(PITHOS_LIBRARY
    NAMES pithos
    PATHS "${PACKAGE_PREFIX_DIR}/lib"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pithos
    REQUIRED_VARS PITHOS_LIBRARY PITHOS_INCLUDE_DIR
)

if(Pithos_FOUND AND NOT TARGET Pithos::pithos)
    add_library(Pithos::pithos UNKNOWN IMPORTED)
    set_target_properties(Pithos::pithos PROPERTIES
        IMPORTED_LOCATION "${PITHOS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${PITHOS_INCLUDE_DIR}"
    )
endif()
