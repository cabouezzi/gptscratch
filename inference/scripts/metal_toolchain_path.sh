#!/bin/sh

set -eu

xcodebuild -showComponent MetalToolchain -json |
  plutil -extract toolchainSearchPath raw -
