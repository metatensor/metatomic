# Converts a file into a list of bytes (with a final NULL terminator), to be
# used as initializer for a C/C++ array:
#
#     const unsigned char DATA[] = {
#     #include "file.inc"
#     };
#
# Usage: cmake -DINPUT_FILE=<input> -DOUTPUT_FILE=<output> -P make_hex_array.cmake

if (NOT CMAKE_SCRIPT_MODE_FILE OR CMAKE_SCRIPT_MODE_DIRECTORY)
    message(FATAL_ERROR "This script is intended to be used with 'cmake -P' and should not be included")
endif()

if (NOT EXISTS "${INPUT_FILE}")
    message(FATAL_ERROR "Input file '${INPUT_FILE}' does not exist")
endif()

file(READ "${INPUT_FILE}" hex HEX)

# Build the list of "0xHH" byte values
string(LENGTH "${hex}" hex_len)
math(EXPR byte_count "${hex_len} / 2")
set(hex_list "")
if (byte_count GREATER 0)
    math(EXPR last_idx "${byte_count} - 1")
    foreach(idx RANGE 0 ${last_idx})
        math(EXPR pos "${idx} * 2")
        string(SUBSTRING "${hex}" ${pos} 2 byte)
        string(TOUPPER "${byte}" byte_upper)
        list(APPEND hex_list "0x${byte_upper}")
    endforeach()
endif()
# NULL terminator, so the array can be used as a C string
list(APPEND hex_list "0x00")

# Format as C initializer, 16 bytes per line
set(output "")
set(line "")
set(line_count 0)
foreach(byte IN LISTS hex_list)
    if (line STREQUAL "")
        set(line "${byte}")
    else()
        set(line "${line}, ${byte}")
    endif()
    math(EXPR line_count "${line_count} + 1")
    math(EXPR mod "${line_count} % 16")
    if (mod EQUAL 0)
        set(output "${output}${line},\n")
        set(line "")
    endif()
endforeach()

if (NOT line STREQUAL "")
    set(output "${output}${line},\n")
endif()

get_filename_component(basename "${INPUT_FILE}" NAME)
file(WRITE "${OUTPUT_FILE}" "/* Generated from ${basename} by make_hex_array.cmake. Do not edit. */\n")
file(APPEND "${OUTPUT_FILE}" "${output}")
