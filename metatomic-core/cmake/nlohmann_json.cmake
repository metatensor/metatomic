# Find or fetch nlohmann JSON library
#
# This module first tries to find nlohmann_json via find_package.
# If that fails, it falls back to fetching it via FetchContent.
#
# After including this module, you can link against nlohmann_json::nlohmann_json

# Guard against multiple inclusion
if(TARGET nlohmann_json::nlohmann_json)
  return()
endif()

if (POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW) # DOWNLOAD_EXTRACT_TIMESTAMP TRUE in FetchContent_Declare
endif()

include(FetchContent)

find_package(nlohmann_json 3.11.0 QUIET)

if(nlohmann_json_FOUND)
  message(STATUS "Found nlohmann_json via find_package: ${nlohmann_json_VERSION}")
else()
  message(STATUS "nlohmann_json not found via find_package, fetching from GitHub")

  # Fetch the release tarball, which contains the CMake build files and headers
  # but not the benchmark reports with very long filenames that break Windows.
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  )

  set(JSON_BuildTests OFF CACHE INTERNAL "")
  # Don't use nlohmann_json's own install rules, they would also install its
  # CMake package config and pkg-config files, and we don't want to advertise
  # a system-wide nlohmann_json package to external users.
  set(JSON_Install OFF CACHE INTERNAL "")

  FetchContent_MakeAvailable(nlohmann_json)

  # nlohmann_json is header-only, so we can install the headers ourselves,
  # alongside metatomic's own (they are used in our public headers).
  install(DIRECTORY "${nlohmann_json_SOURCE_DIR}/include/nlohmann"
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
