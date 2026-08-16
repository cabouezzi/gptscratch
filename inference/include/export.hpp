#pragma once

#if defined _WIN32 || defined __CYGWIN__
  #ifdef BUILDING_INFERENCE
    #define INFERENCE_PUBLIC __declspec(dllexport)
  #else
    #define INFERENCE_PUBLIC __declspec(dllimport)
  #endif
#elif defined __OS2__
  #ifdef BUILDING_INFERENCE
    #define INFERENCE_PUBLIC __declspec(dllexport)
  #else
    #define INFERENCE_PUBLIC
  #endif
#else
  #ifdef BUILDING_INFERENCE
    #define INFERENCE_PUBLIC __attribute__((visibility("default")))
  #else
    #define INFERENCE_PUBLIC
  #endif
#endif
