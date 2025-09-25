/******************************************************************************
 * Copyright (c) 2025, xxx.
 ******************************************************************************/

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_ALIBI
  #define ALIBI_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    if (HEADDIM <= 32) {                   \
      constexpr static int kHeadDim = 32;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 64) {            \
      constexpr static int kHeadDim = 64;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 96) {            \
      constexpr static int kHeadDim = 96;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 128) {           \
      constexpr static int kHeadDim = 128; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 160) {           \
      constexpr static int kHeadDim = 160; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 192) {           \
      constexpr static int kHeadDim = 192; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 256) {           \
      constexpr static int kHeadDim = 256; \
      return __VA_ARGS__();                \
    }                                      \
  }()

  
// New macro for d_rounded
#define HEADDIM_ROUNDED_SWITCH(D_ROUNDED_VAL, CONST_NAME, ...) \
  [&] {                                                        \
    if (D_ROUNDED_VAL <= 32) {                                 \
      constexpr static int CONST_NAME = 32;                    \
      return __VA_ARGS__();                                    \
    } else if (D_ROUNDED_VAL <= 64) {                          \
      constexpr static int CONST_NAME = 64;                    \
      return __VA_ARGS__();                                    \
    } else if (D_ROUNDED_VAL <= 96) {                          \
      constexpr static int CONST_NAME = 96;                    \
      return __VA_ARGS__();                                    \
    } else if (D_ROUNDED_VAL <= 128) {                         \
      constexpr static int CONST_NAME = 128;                   \
      return __VA_ARGS__();                                    \
    } else if (D_ROUNDED_VAL <= 160) {                         \
      constexpr static int CONST_NAME = 160;                   \
      return __VA_ARGS__();                                    \
    } else if (D_ROUNDED_VAL <= 192) {                         \
      constexpr static int CONST_NAME = 192;                   \
      return __VA_ARGS__();                                    \
    } else if (D_ROUNDED_VAL <= 256) {                         \
      constexpr static int CONST_NAME = 256;                   \
      return __VA_ARGS__();                                    \
    } else {                                                   \
      TORCH_CHECK(false, "Unsupported D_ROUNDED_VAL for HEADDIM_ROUNDED_SWITCH: ", D_ROUNDED_VAL); \
      constexpr static int CONST_NAME = 0; /* Should not reach */ \
      return __VA_ARGS__();                                    \
    }                                                          \
  }()
