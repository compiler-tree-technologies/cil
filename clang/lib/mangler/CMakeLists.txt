add_clang_library(CILMangle
  CILMangle.cpp
)

target_link_libraries(CILMangle
  clangAST
)

install(TARGETS CILMangle DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)