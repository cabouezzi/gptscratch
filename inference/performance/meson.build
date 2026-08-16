matmul_performance_sources = files('cpu/matmul.cpp') + matmul_source + metal_backend_source

performance_metal_artifacts = custom_target(
  'performance-metallib',
  output : 'inference.metallib',
  command : [metal_compiler, metal_toolchain_path, metal_source_directory, '@OUTPUT@'],
  build_always_stale : true,
)
performance_metallib = performance_metal_artifacts

executable(
  'matmul_performance',
  matmul_performance_sources,
  include_directories : [project_includes, metal_cpp_includes],
  dependencies : dependencies,
  link_depends : performance_metallib,
  install : false,
)
