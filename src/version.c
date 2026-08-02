/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* version.c — what the linked library actually is. */

#include "waste.h"
#include "crc32.h"
#include "waste_backend.h"
#include "waste_format.h"

#include <stdio.h>

const char *waste_version(void) { return WASTE_VERSION_STRING; }
int waste_version_number(void) { return WASTE_VERSION_NUMBER; }
int waste_api_version(void) { return WASTE_API_VERSION; }
size_t waste_sizeof_cfg(void) { return sizeof(waste_cfg); }
size_t waste_sizeof_memplan(void) { return sizeof(waste_memplan); }

const char *waste_build_info(void)
{
    static char buf[192];
    static int done = 0;
    if (!done) {
        waste_backend_init(WASTE_BE_AUTO);
        /* The crc32 implementation is in here because it is chosen by
         * compiler flags rather than at run time, and the two differ by
         * 12x on the read path — so which one a build got is not
         * something to guess at from the outside. */
        snprintf(buf, sizeof buf,
                 "WASTE %s (upstream %s, container v%d, backend %s, crc32 %s, %s)",
                 WASTE_VERSION_STRING, WASTE_UPSTREAM_VERSION_STRING,
                 WASTE_FORMAT_VERSION,
                 waste_backend_name(), waste_crc32_impl(),
#if defined(__aarch64__)
                 "arm64"
#elif defined(__x86_64__)
                 "x86_64"
#else
                 "generic"
#endif
        );
        done = 1;
    }
    return buf;
}
