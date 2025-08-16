// Stub to provide missing _glapi_tls_Current symbol
#include <pthread.h>

// Provide the missing TLS symbol
__thread void *_glapi_tls_Current = 0;