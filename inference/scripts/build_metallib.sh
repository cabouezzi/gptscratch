#!/bin/bash

set -eu

toolchain_root="$1"
shader_directory="$2"
metallib_output="$3"

metal="${toolchain_root}/Metal.xctoolchain/usr/bin/metal"
air_directory="$(dirname "${metallib_output}")/.metal-air"

shopt -s nullglob
shader_sources=("${shader_directory}"/*.metal)

if [ "${#shader_sources[@]}" -eq 0 ]; then
  echo "No .metal files found in ${shader_directory}" >&2
  exit 1
fi

needs_rebuild=false
if [ ! -f "${metallib_output}" ] || [ "$0" -nt "${metallib_output}" ]; then
  needs_rebuild=true
fi

for shader_source in "${shader_sources[@]}"; do
  if [ "${shader_source}" -nt "${metallib_output}" ]; then
    needs_rebuild=true
  fi
done

if [ "${needs_rebuild}" = false ]; then
  exit 0
fi

mkdir -p "${air_directory}"
air_outputs=()
for shader_source in "${shader_sources[@]}"; do
  shader_name="$(basename "${shader_source}" .metal)"
  air_output="${air_directory}/${shader_name}.air"
  "${metal}" -c "${shader_source}" -o "${air_output}"
  air_outputs+=("${air_output}")
done

"${metal}" -o "${metallib_output}" "${air_outputs[@]}"
