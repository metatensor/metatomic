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

include(FetchContent)

# First, try to find nlohmann_json if it's already installed on the system
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
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )

  set(JSON_BuildTests OFF CACHE INTERNAL "")
  set(JSON_Install ON CACHE INTERNAL "")

  FetchContent_MakeAvailable(nlohmann_json)

  message(STATUS "nlohmann_json fetched successfully")
endif()
