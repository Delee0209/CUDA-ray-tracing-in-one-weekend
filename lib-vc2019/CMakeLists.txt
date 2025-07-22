message("-- Linking GLFW3")
add_library(GLFW3 glfw3.dll)
set_target_properties(GLFW3 PROPERTIES LINKER_LANGUAGE C)