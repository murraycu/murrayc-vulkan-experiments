# Put compiled object files into the same sub-directory as their sources
AUTOMAKE_OPTIONS = subdir-objects

# Point to our macro directory and pick up user flags from the environment
ACLOCAL_AMFLAGS  = -I macros ${ACLOCAL_FLAGS} 
# Make warnings always fatal at distcheck time, and disable execution
# of tools that change files not installed by the package itself
DISTCHECK_CONFIGURE_FLAGS = --enable-warnings=fatal --disable-update-mime-database

dist_noinst_SCRIPTS = autogen.sh

# Put the CMake build files in the tarball created by autotools' "make distcheck".
dist_noinst_DATA = CMakeLists.txt \
		   tests/CMakeLists.txt

bin_PROGRAMS = example

example_SOURCES = src/main.cc \
		  tinyobjloader/tiny_obj_loader.h
example_CXXFLAGS = ${PROJECT_CFLAGS} ${PROJECT_WXXFLAGS}
example_LDADD = ${PROJECT_LIBS}

# Build googletest:
gtest_top = googletest/googletest
gtest_top_srcdir = $(top_srcdir)/$(gtest_top)
check_LTLIBRARIES = libgtest.la

# TODO: Use the gtest_top variable when we know we are always using automake >= 1.16
# ( See the mention of bug 13928 in https://lists.gnu.org/archive/html/info-gnu/2018-02/msg00008.html )
# Otherwise, the build will fail because it looks for .Plo files in a deps/ directory in a path that mentions
# the actual variable name, not the variable's expanded value.
#
# libgtest_la_SOURCES = $(gtest_top)/src/gtest-all.cc \
#		      $(gtest_top)/src/gtest_main.cc

libgtest_la_SOURCES = googletest/googletest/src/gtest-all.cc \
		      googletest/googletest/src/gtest_main.cc

libgtest_la_CPPFLAGS = -I$(gtest_top_srcdir)/include -I$(gtest_top_srcdir) \
		       $(PROJECT_GTEST_WXXFLAGS)
libgtest_la_LDFLAGS = -pthread
EXTRA_DIST = $(gtest_top_srcdir)/src/gtest-all.cc \
	     $(gtest_top_srcdir)/src/gtest-filepath.cc \
	     $(gtest_top_srcdir)/src/gtest-printers.cc \
	     $(gtest_top_srcdir)/src/gtest.cc \
	     $(gtest_top_srcdir)/src/gtest_main.cc \
	     $(gtest_top_srcdir)/src/gtest-matchers.cc \
	     $(gtest_top_srcdir)/src/gtest-test-part.cc \
	     $(gtest_top_srcdir)/src/gtest-death-test.cc \
	     $(gtest_top_srcdir)/src/gtest-port.cc \
	     $(gtest_top_srcdir)/src/gtest-typed-test.cc \
	     $(gtest_top_srcdir)/src/gtest-internal-inl.h \
	     $(gtest_top_srcdir)/include/gtest/gtest-death-test.h \
	     $(gtest_top_srcdir)/include/gtest/gtest_pred_impl.h \
	     $(gtest_top_srcdir)/include/gtest/gtest-test-part.h \
	     $(gtest_top_srcdir)/include/gtest/gtest.h \
	     $(gtest_top_srcdir)/include/gtest/gtest-printers.h \
	     $(gtest_top_srcdir)/include/gtest/gtest-typed-test.h \
	     $(gtest_top_srcdir)/include/gtest/gtest-matchers.h \
	     $(gtest_top_srcdir)/include/gtest/gtest-message.h \
	     $(gtest_top_srcdir)/include/gtest/gtest_prod.h \
	     $(gtest_top_srcdir)/include/gtest/gtest-param-test.h \
	     $(gtest_top_srcdir)/include/gtest/gtest-spi.h \
	     $(gtest_top_srcdir)/include/gtest/internal/gtest-death-test-internal.h \
	     $(gtest_top_srcdir)/include/gtest/internal/gtest-port-arch.h \
	     $(gtest_top_srcdir)/include/gtest/internal/gtest-filepath.h \
	     $(gtest_top_srcdir)/include/gtest/internal/gtest-port.h \
	     $(gtest_top_srcdir)/include/gtest/internal/gtest-internal.h \
	     $(gtest_top_srcdir)/include/gtest/internal/gtest-string.h \
	     $(gtest_top_srcdir)/include/gtest/internal/gtest-type-util.h \
	     $(gtest_top_srcdir)/include/gtest/internal/gtest-param-util.h \
	     $(gtest_top_srcdir)/include/gtest/internal/custom/gtest.h \
	     $(gtest_top_srcdir)/include/gtest/internal/custom/gtest-printers.h \
	     $(gtest_top_srcdir)/include/gtest/internal/custom/gtest-port.h

COMMON_CXXFLAGS = $(PROJECT_CFLAGS) $(PROJECT_WXXFLAGS)
COMMON_CPPFLAGS = -I$(top_builddir)/src -I$(top_srcdir)/src \
		  -isystem $(gtest_top_srcdir)/include
COMMON_LIBS = $(PROJECT_LIBS) libgtest.la

check_PROGRAMS = \
  tests/test_thing

TESTS = $(check_PROGRAMS)

#List of source files needed to build the executable:

tests_test_thing_SOURCES = \
	tests/test_thing.cc
tests_test_thing_CPPFLAGS = \
	$(COMMON_CPPFLAGS)
tests_test_thing_CXXFLAGS = \
	$(COMMON_CXXFLAGS)
tests_test_thing_LDADD = \
	$(COMMON_LIBS)


CLEANFILES = stamp-h2


format:
	clang-format -i src/*.cc src/shaders/*.vert src/shaders/*.frag
