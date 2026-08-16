project(
  'inference',
  'cpp',
  version : '0.1',
  meson_version : '>= 1.3.0',
  default_options : ['warning_level=3', 'cpp_std=c++23'],
)

# macOS system frameworks
apple_frameworks = dependency(
  'appleframeworks',
  modules : ['Metal', 'Foundation', 'QuartzCore'],
)

dependencies = [apple_frameworks]
catch2_dep = dependency(
  'catch2',
  fallback : ['catch2', 'catch2_dep'],
)
project_includes = include_directories('include')
metal_cpp_includes = include_directories('third_party/metal-cpp')
metal_compiler = find_program('scripts/build_metallib.sh')
metal_source_directory = join_paths(meson.project_source_root(), 'src', 'metal')
metal_toolchain_path = run_command(
  find_program('scripts/metal_toolchain_path.sh'),
  check : true,
).stdout().strip()

metal_artifacts = custom_target(
  'inference-metallib',
  output : 'inference.metallib',
  command : [metal_compiler, metal_toolchain_path, metal_source_directory, '@OUTPUT@'],
  build_always_stale : true,
)
inference_metallib = metal_artifacts

# These arguments are only used to build the shared library
# not the executables that use the library.
lib_args = ['-DBUILDING_INFERENCE']

matmul_source = files('src/cpu/matmul.cpp')
metal_backend_source = files('src/metal/backend.cpp')
sources = files(
  'src/model.cpp',
  'src/tokenizer.cpp',
) + matmul_source + metal_backend_source

lib = library(
  'inference',
  sources,
  install : true,
  cpp_shared_args : lib_args,
  gnu_symbol_visibility : 'hidden',
  include_directories : [project_includes, metal_cpp_includes],
  dependencies : dependencies,
)

app_exe = executable(
  'inference_app',
  'inference_app.cpp',
  include_directories : project_includes,
  dependencies : dependencies,
  link_with : lib,
  link_depends : inference_metallib,
)

test_exe = executable(
  'inference_test',
  [
    'inference_test.cpp',
    'tests/cpu/matmul.cpp',
    'tests/metal/matmul.cpp',
    'src/cpu/matmul.cpp',
    'src/metal/backend.cpp',
  ],
  include_directories : [project_includes, metal_cpp_includes],
  dependencies : dependencies + [catch2_dep],
  link_with : lib,
  link_depends : inference_metallib,
)
test('inference', test_exe)

subdir('performance')

# Make this library usable as a Meson subproject.
inference_dep = declare_dependency(
  include_directories : project_includes,
  dependencies : dependencies,
  link_with : lib,
)
meson.override_dependency('inference', inference_dep)

# Make this library usable from the system's
# package manager.
install_headers(
  'include/export.hpp',
  'include/inference.hpp',
  'include/metal/backend.hpp',
  'include/model.hpp',
  'include/tokenizer.hpp',
)

pkg_mod = import('pkgconfig')
pkg_mod.generate(
  lib,
  description : 'Meson sample project.',
)
