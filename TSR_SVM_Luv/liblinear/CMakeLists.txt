
file(GLOB_RECURSE SRC *.c *.cpp *.cc)
file(GLOB_RECURSE header *.h *.hpp)
add_library( liblinear STATIC ${SRC} ${header})
#target_link_libraries( liblinear) 无链接库
