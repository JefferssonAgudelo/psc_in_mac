
#include "ggcm_mhd_private.h"
#include "ggcm_mhd_defs.h"

#include <mrc_fld_as_double.h>

#define MT MT_BGRID_FC
#define BGRID_SFX(x) x ## _bgrid_fc

#include "ggcm_mhd_bgrid_common.c"
