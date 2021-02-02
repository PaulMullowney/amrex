set(_sources     main.cpp )
set(_input_files inputs  )
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

setup_test(_sources _input_files)

unset(_sources)
unset(_input_files)
