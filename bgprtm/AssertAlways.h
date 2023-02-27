#ifndef ASSERT_ALWAYS_H_
#define ASSERT_ALWAYS_H_

/* make sure ENSURE_DEBUG isn't already defined */
#ifdef ENSURE_NDEBUG
#undef ENSURE_NDEBUG
#endif

/* move NDEBUG setting to a safe spot for a moment */
#ifdef NDEBUG
#define ENSURE_NDEBUG
#undef NDEBUG
#endif

/* Now assert() should be defined */
#include <assert.h>

# define assertAlwaysPrint(expr, value) \
  (__ASSERT_VOID_CAST ((expr) ? 0 :               \
                       (fprintf(stderr,"Assert value: %.12g \n", (double)(value)  ) )));    \
  (__ASSERT_VOID_CAST ((expr) ? 0 :               \
                       (__assert_fail (__STRING(expr), __FILE__, __LINE__,    \
                                       __ASSERT_FUNCTION), 0)))

# define assertAlways(expr) \
  (__ASSERT_VOID_CAST ((expr) ? 0 :               \
                       (__assert_fail (__STRING(expr), __FILE__, __LINE__,    \
                                       __ASSERT_FUNCTION), 0)))

/* bring back NDEBUG, if set earlier */
#ifdef ENSURE_NDEBUG
#undef ENSURE_NDEBUG
#define NDEBUG
#undef assert
#define assert(expr)    (__ASSERT_VOID_CAST (0))
#undef assert_perror
#define assert_perror(errnum) (__ASSERT_VOID_CAST (0))

#endif


#endif /*ENSURE_H_*/
                    
