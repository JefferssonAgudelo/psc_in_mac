
#include "ggcm_mhd_defs.h"
#include "ggcm_mhd_private.h"
#include "ggcm_mhd_crds.h"
#include "ggcm_mhd_diag.h"
#include "ggcm_mhd_ic_private.h"

#include <mrc_fld.h>
#include <mrc_fld_as_float.h>
#include <mrc_domain.h>

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h> 
#include <assert.h>

// ======================================================================
// ggcm_mhd_ic subclass "cosine"

struct ggcm_mhd_ic_cosine {
  //float l_x;
  float by0; 
  float bz0;
  float rho0; 
  float t0; 
  float tau;   
  //  float xs;
  float pert;
};

// ----------------------------------------------------------------------
// ggcm_mhd_ic_cosine_run

static void
ggcm_mhd_ic_cosine_run(struct ggcm_mhd_ic *ic)
{
  struct ggcm_mhd_ic_cosine *sub = mrc_to_subobj(ic, struct ggcm_mhd_ic_cosine);
  struct ggcm_mhd *mhd = ic->mhd;
  struct mrc_fld *f3 = mrc_fld_get_as(mhd->fld, FLD_TYPE);
  struct mrc_crds *crds = mrc_domain_get_crds(mhd->domain);  

  struct mrc_fld *fld_psi = mrc_domain_fld_create(mhd->domain, SW_2, NULL);
  mrc_fld_set_type(fld_psi, FLD_TYPE);
  mrc_fld_setup(fld_psi);

  const double *lo = mrc_crds_lo(crds), *hi = mrc_crds_hi(crds);
  float L[3], r[3];
  for(int i = 0; i < 3; i++){
    L[i] = hi[i] - lo[i];
  }
  float kx = 2.*M_PI / L[0] ; 
  float ky = 2.*M_PI / L[1] ; 

  //float l_x = sub->l_x; // Scale length of the magenetic shear at each surface
  float by0 = sub->by0; // Asymptotic reconnection field
  float bz0 = sub->bz0; // Uniform guide field
  float rho0 = sub->rho0; // Asymptotic plasma density
  //float t0 = sub->t0; // Uniform electron temperature (Te)
  //float tau = sub->tau; // Ratio of ion temperature to electron temperature (Ti/Te)
  //float xs = sub->xs; // Distance of each current sheet from zero
  //float pert = sub->pert; // 

  for (int p = 0; p < mrc_fld_nr_patches(f3); p++) {
    mrc_fld_foreach(f3, ix, iy, iz, 2, 2) {
      r[0] = .5*(MRC_CRDX(crds, ix) + MRC_CRDX(crds, ix-1));
      r[1] = .5*(MRC_CRDY(crds, iy) + MRC_CRDY(crds, iy-1));
      
      M3(fld_psi, 0, ix,iy,iz, p) = ( L[0] / (4. * M_PI) ) * (1. - cos(2*kx*r[0])) * sin(ky*r[1]);   
      //M3(fld_psi, 0, ix,iy,iz, p) = 0.01*cos(kx*r[0])*(sin(ky*r[1]));
      //(L[0] / (4.*M_PI)) * ((1 - cos(kx*r[0])) * sin(ky*r[1]));
    } mrc_fld_foreach_end;
  }

  float *bd2x = ggcm_mhd_crds_get_crd(mhd->crds, 0, BD2);
  //float *bd2y = ggcm_mhd_crds_get_crd(mhd->crds, 1, BD2);

  for (int p = 0; p < mrc_fld_nr_patches(f3); p++) {
    mrc_fld_foreach(f3, ix, iy, iz, 1, 1) {
      // FIXME! the staggering for B is okay, but fld_psi and other stuff below needs to be
      // fixed / checked for cell-centered
      r[0] = MRC_MCRD(crds, 0, ix, p);
      r[1] = MRC_MCRD(crds, 1, iy, p);
      r[2] = MRC_MCRD(crds, 2, iz, p); 
    
      BY_(f3, ix,iy,iz, p) = by0 * cos(kx*r[0]) -
	(M3(fld_psi, 0, ix+1,iy,iz, p) - M3(fld_psi, 0, ix,iy,iz, p)) / bd2x[ix];
      BX_(f3, ix,iy,iz, p) =  0.0;
      //(M3(fld_psi, 0, ix,iy+1,iz, p) - M3(fld_psi, 0, ix,iy,iz, p)) / bd2y[iy];
      BZ_(f3, ix,iy,iz, p) = sqrt( sqr(bz0) - sqr(BY_(f3,ix,iy,iz, p)) );
      RR_(f3, ix,iy,iz, p) = rho0;
      PP_(f3, ix,iy,iz, p) = RR_(f3, ix,iy,iz, p);
    } mrc_fld_foreach_end;
  }

  mrc_fld_put_as(f3, mhd->fld);
  mrc_fld_destroy(fld_psi);

  ggcm_mhd_convert_from_primitive(mhd, mhd->fld);
}

// ----------------------------------------------------------------------
// ggcm_mhd_ic_cosine_descr

#define VAR(x) (void *)offsetof(struct ggcm_mhd_ic_cosine, x)
static struct param ggcm_mhd_ic_cosine_descr[] = {
  //{"l_b", VAR(l_b), PARAM_FLOAT(0.1)},
  {"by0", VAR(by0), PARAM_FLOAT(1.0)},
  {"bz0", VAR(bz0), PARAM_FLOAT(1.0)},
  {"rho0", VAR(rho0), PARAM_FLOAT(1.0)},  
  {"t0", VAR(t0), PARAM_FLOAT(0.5)},  
  {"tau", VAR(tau), PARAM_FLOAT(1.0)},
  {"pert",VAR(pert), PARAM_FLOAT(1.0)},
  {},
};
#undef VAR

// ----------------------------------------------------------------------
// ggcm_mhd_ic_cosine_ops

struct ggcm_mhd_ic_ops ggcm_mhd_ic_cosine_ops = {
  .name        = "cosine",
  .size        = sizeof(struct ggcm_mhd_ic_cosine),
  .param_descr = ggcm_mhd_ic_cosine_descr,
  .run         = ggcm_mhd_ic_cosine_run,
};


// ======================================================================
// ggcm_mhd class "cosine"

// ----------------------------------------------------------------------
// ggcm_mhd_cosine_create

static void
ggcm_mhd_cosine_create(struct ggcm_mhd *mhd)
{
  ggcm_mhd_default_box(mhd);

  /* set defaults for coord arrays */
  struct mrc_crds *crds = mrc_domain_get_crds(mhd->domain);
  mrc_crds_set_type(crds, "two_gaussian");
  mrc_crds_set_param_int(crds, "sw", SW_2);   // 'stencil width' 
  mrc_crds_set_param_double3(crds, "l", (double[3]) {  0.0, 0.0, -1.0 });
  mrc_crds_set_param_double3(crds, "h", (double[3]) {  2.*M_PI, 2.*M_PI,  1.0 });
}

static struct ggcm_mhd_ops ggcm_mhd_cosine_ops = {
  .name             = "cosine",
  .create           = ggcm_mhd_cosine_create,
};

// ======================================================================

extern struct ggcm_mhd_diag_ops ggcm_mhd_diag_c_ops;

int
main(int argc, char **argv)
{
  mrc_class_register_subclass(&mrc_class_ggcm_mhd, &ggcm_mhd_cosine_ops);  
  mrc_class_register_subclass(&mrc_class_ggcm_mhd_diag, &ggcm_mhd_diag_c_ops);
  mrc_class_register_subclass(&mrc_class_ggcm_mhd_ic, &ggcm_mhd_ic_cosine_ops);  
 
  return ggcm_mhd_main(&argc, &argv);
}

