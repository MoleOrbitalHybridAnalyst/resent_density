/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Authors: Qiming Sun <osirpt.sun@gmail.com>
 *          Susi Lehtola <susi.lehtola@gmail.com>
 *          Xing Zhang <zhangxing.nju@gmail.com>
 *
 * libxc from
 * http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:manual
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <xc.h>
#include "config.h"
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))


static void _eval_xc_lda(xc_func_type *func_x, int spin, int np,
                         double* rho_u, double* rho_d,
                         double *ex, double *vxc, double *fxc, double *kxc)
{
        int i;
        double *rho;
        if (spin == XC_POLARIZED) {
                rho = malloc(sizeof(double) * np*2);
                for (i = 0; i < np; i++) {
                        rho[i*2+0] = rho_u[i];
                        rho[i*2+1] = rho_d[i];
                }
        } else {
                rho = rho_u;
        }

        // rho, vxc, fxc, kxc
        const int seg0[] = {1,1,1,1};
        const int seg1[] = {2,2,3,4};
        const int *seg;
        if (spin == XC_POLARIZED) {
            seg = seg1;
        } else {
            seg = seg0;
        }

#pragma omp parallel
{
        int nblk = omp_get_num_threads();
        if (np < nblk) {nblk = 1;}
        int blk_size = np / nblk;

        int iblk;
        double *prho, *pex, *pvxc=NULL, *pfxc=NULL, *pkxc=NULL;
        #pragma omp for schedule(static)
        for (iblk = 0; iblk < nblk; iblk++) {
                prho = rho + iblk * blk_size * seg[0];
                pex = ex + iblk * blk_size;
                if (vxc != NULL) {
                        pvxc = vxc + iblk * blk_size * seg[1];
                }
                if (fxc != NULL) {
                        pfxc = fxc + iblk * blk_size * seg[2];
                }
                if (kxc != NULL) {
                        pkxc = kxc + iblk * blk_size * seg[3];
                }
#if XC_MAJOR_VERSION >= 5
                xc_lda_exc_vxc_fxc_kxc(func_x, blk_size, prho, pex, pvxc, pfxc, pkxc);
#else
                xc_lda(func_x, blk_size, prho, pex, pvxc, pfxc, pkxc);
#endif
        }

#pragma omp single
{
        int np_res = np - nblk * blk_size;
        if (np_res > 0) {
                prho = rho + nblk * blk_size * seg[0];
                pex = ex + nblk * blk_size;
                if (vxc != NULL) {
                        pvxc = vxc + nblk * blk_size * seg[1];
                }
                if (fxc != NULL) {
                        pfxc = fxc + nblk * blk_size * seg[2];
                }
                if (kxc != NULL) {
                        pkxc = kxc + nblk * blk_size * seg[3];
                }
#if XC_MAJOR_VERSION >= 5
                xc_lda_exc_vxc_fxc_kxc(func_x, np_res, prho, pex, pvxc, pfxc, pkxc);
#else
                xc_lda(func_x, np_res, prho, pex, pvxc, pfxc, pkxc);
#endif
        }
} // omp single
} // omp parallel

        if (spin == XC_POLARIZED) {
            free(rho);
        }
}


static void _eval_xc_gga(xc_func_type *func_x, int spin, int np,
                         double* rho_u, double* rho_d,
                         double *ex, double *vxc, double *fxc, double *kxc)
{
        int i;
        double *rho, *sigma;
        double *gxu, *gyu, *gzu, *gxd, *gyd, *gzd;
        double *vsigma = NULL;
        double *v2rhosigma  = NULL;
        double *v2sigma2    = NULL;
        double *v3rho2sigma = NULL;
        double *v3rhosigma2 = NULL;
        double *v3sigma3    = NULL;

        if (spin == XC_POLARIZED) {
                rho = malloc(sizeof(double) * np*2);
                sigma = malloc(sizeof(double) * np*3);
                gxu = rho_u + np;
                gyu = rho_u + np * 2;
                gzu = rho_u + np * 3;
                gxd = rho_d + np;
                gyd = rho_d + np * 2;
                gzd = rho_d + np * 3;
                for (i = 0; i < np; i++) {
                        rho[i*2+0] = rho_u[i];
                        rho[i*2+1] = rho_d[i];
                        sigma[i*3+0] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                        sigma[i*3+1] = gxu[i]*gxd[i] + gyu[i]*gyd[i] + gzu[i]*gzd[i];
                        sigma[i*3+2] = gxd[i]*gxd[i] + gyd[i]*gyd[i] + gzd[i]*gzd[i];
                }
        } else {
                rho = rho_u;
                sigma = malloc(sizeof(double) * np);
                gxu = rho_u + np;
                gyu = rho_u + np * 2;
                gzu = rho_u + np * 3;
                for (i = 0; i < np; i++) {
                        sigma[i] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                }
        }

        // rho, sigma
        const int seg0[] = {1,1};
        const int seg1[] = {2,3};
        // vrho, vsigma
        const int vseg0[] = {1,1};
        const int vseg1[] = {2,3};
        // v2rho2, v2rhosigma, v2sigma2
        const int fseg0[] = {1,1,1};
        const int fseg1[] = {3,6,6};
        // v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3
        const int kseg0[] = {1,1,1,1};
        const int kseg1[] = {4,9,12,10};

        const int *seg, *vseg, *fseg, *kseg;
        if (spin == XC_POLARIZED) {
                seg = seg1;
                vseg = vseg1;
                fseg = fseg1;
                kseg = kseg1;
        } else {
                seg = seg0;
                vseg = vseg0;
                fseg = fseg0;
                kseg = kseg0;
        }

        if (vxc != NULL) {
                // vrho = vxc
                vsigma = vxc + np * vseg[0];
        }
        if (fxc != NULL) {
                // v2rho2 = fxc
                v2rhosigma = fxc + np * fseg[0];
                v2sigma2 = v2rhosigma + np * fseg[1];
        }
        if (kxc != NULL) {
                // v3rho3 = kxc
                v3rho2sigma = kxc + np * kseg[0];
                v3rhosigma2 = v3rho2sigma + np * kseg[1];
                v3sigma3 = v3rhosigma2 + np * kseg[2];
        }

#pragma omp parallel
{
        int nblk = omp_get_num_threads();
        if (np < nblk) {nblk = 1;}
        int blk_size = np / nblk;

        int iblk;
        double *prho, *psigma, *pex;
        double *pvxc=NULL, *pvsigma=NULL;
        double *pfxc=NULL, *pv2rhosigma=NULL, *pv2sigma2=NULL;
        double *pkxc=NULL, *pv3rho2sigma=NULL, *pv3rhosigma2=NULL, *pv3sigma3=NULL;
        #pragma omp for schedule(static)
        for (iblk = 0; iblk < nblk; iblk++) {
                prho = rho + iblk * blk_size * seg[0];
                psigma = sigma + iblk * blk_size * seg[1];
                pex = ex + iblk * blk_size;
                if (vxc != NULL) {
                    pvxc = vxc + iblk * blk_size * vseg[0]; 
                    pvsigma = vsigma + iblk * blk_size * vseg[1]; 
                }
                if (fxc != NULL) {
                    pfxc = fxc + iblk * blk_size * fseg[0];
                    pv2rhosigma = v2rhosigma + iblk * blk_size * fseg[1];
                    pv2sigma2 = v2sigma2 + iblk * blk_size * fseg[2];
                }
                if (kxc != NULL) {
                    pkxc = kxc + iblk * blk_size * kseg[0];
                    pv3rho2sigma = v3rho2sigma + iblk * blk_size * kseg[1];
                    pv3rhosigma2 = v3rhosigma2 + iblk * blk_size * kseg[2];
                    pv3sigma3 = v3sigma3 + iblk * blk_size * kseg[3];
                }
#if (XC_MAJOR_VERSION == 2 && XC_MINOR_VERSION < 2)
                xc_gga(func_x, blk_size, prho, psigma, pex,
                       pvxc, pvsigma, pfxc, pv2rhosigma, pv2sigma2);
#elif XC_MAJOR_VERSION < 5
                xc_gga(func_x, blk_size, prho, psigma, pex,
                       pvxc, pvsigma, pfxc, pv2rhosigma, pv2sigma2,
                       pkxc, pv3rho2sigma, pv3rhosigma2, pv3sigma3);
#else
                xc_gga_exc_vxc_fxc_kxc(func_x, blk_size, prho, psigma, pex,
                       pvxc, pvsigma, pfxc, pv2rhosigma, pv2sigma2,
                       pkxc, pv3rho2sigma, pv3rhosigma2, pv3sigma3);
#endif
        }


#pragma omp single
{
        int np_res = np - nblk * blk_size;
        if (np_res > 0) {
                prho = rho + nblk * blk_size * seg[0];
                psigma = sigma + nblk * blk_size * seg[1];
                pex = ex + nblk * blk_size;
                if (vxc != NULL) {
                    pvxc = vxc + nblk * blk_size * vseg[0];
                    pvsigma = vsigma + nblk * blk_size * vseg[1];
                }
                if (fxc != NULL) {
                    pfxc = fxc + nblk * blk_size * fseg[0];
                    pv2rhosigma = v2rhosigma + nblk * blk_size * fseg[1];
                    pv2sigma2 = v2sigma2 + nblk * blk_size * fseg[2];
                }
                if (kxc != NULL) {
                    pkxc = kxc + nblk * blk_size * kseg[0];
                    pv3rho2sigma = v3rho2sigma + nblk * blk_size * kseg[1];
                    pv3rhosigma2 = v3rhosigma2 + nblk * blk_size * kseg[2];
                    pv3sigma3 = v3sigma3 + nblk * blk_size * kseg[3];
                }
#if (XC_MAJOR_VERSION == 2 && XC_MINOR_VERSION < 2)
                xc_gga(func_x, np_res, prho, psigma, pex,
                       pvxc, pvsigma, pfxc, pv2rhosigma, pv2sigma2);
#elif XC_MAJOR_VERSION < 5
                xc_gga(func_x, np_res, prho, psigma, pex,
                       pvxc, pvsigma, pfxc, pv2rhosigma, pv2sigma2,
                       pkxc, pv3rho2sigma, pv3rhosigma2, pv3sigma3);
#else
                xc_gga_exc_vxc_fxc_kxc(func_x, np_res, prho, psigma, pex,
                       pvxc, pvsigma, pfxc, pv2rhosigma, pv2sigma2,
                       pkxc, pv3rho2sigma, pv3rhosigma2, pv3sigma3);
#endif
        }
} // omp single
} // omp parallel

        if (spin == XC_POLARIZED) {
                free(rho);
        }
        free(sigma);
}


static void _eval_xc_mgga(xc_func_type *func_x, int spin, int np,
                          double* rho_u, double* rho_d,
                          double *ex, double *vxc, double *fxc, double *kxc)
{
        int i;
        double *rho, *sigma, *lapl, *tau;
        double *gxu, *gyu, *gzu, *gxd, *gyd, *gzd;
        double *lapl_u, *lapl_d, *tau_u, *tau_d;

        if (spin == XC_POLARIZED) {
                rho = malloc(sizeof(double) * np*2);
                sigma = malloc(sizeof(double) * np*3);
                lapl = malloc(sizeof(double) * np*2);
                tau = malloc(sizeof(double) * np*2);
                gxu = rho_u + np;
                gyu = rho_u + np * 2;
                gzu = rho_u + np * 3;
                gxd = rho_d + np;
                gyd = rho_d + np * 2;
                gzd = rho_d + np * 3;
                lapl_u = rho_u + np * 4;
                tau_u  = rho_u + np * 5;
                lapl_d = rho_d + np * 4;
                tau_d  = rho_d + np * 5;
                for (i = 0; i < np; i++) {
                        rho[i*2+0] = rho_u[i];
                        rho[i*2+1] = rho_d[i];
                        sigma[i*3+0] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                        sigma[i*3+1] = gxu[i]*gxd[i] + gyu[i]*gyd[i] + gzu[i]*gzd[i];
                        sigma[i*3+2] = gxd[i]*gxd[i] + gyd[i]*gyd[i] + gzd[i]*gzd[i];
                        lapl[i*2+0] = lapl_u[i];
                        lapl[i*2+1] = lapl_d[i];
                        tau[i*2+0] = tau_u[i];
                        tau[i*2+1] = tau_d[i];
                }
        } else {
                rho = rho_u;
                sigma = malloc(sizeof(double) * np);
                lapl = rho_u + np * 4;
                tau  = rho_u + np * 5;
                gxu = rho_u + np;
                gyu = rho_u + np * 2;
                gzu = rho_u + np * 3;
                for (i = 0; i < np; i++) {
                        sigma[i] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                }
        }

        // rho, sigma, lapl, tau
        const int seg0[] = {1,1,1,1};
        const int seg1[] = {2,3,2,2};
        // vrho, vsigma, vlapl, vtau
        const int vseg0[] = {1,1,1,1};
        const int vseg1[] = {2,3,2,2};
        // v2rho2, v2rhosigma, v2sigma2, v2lapl2, v2tau2,
        // v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau
        const int fseg0[] = {1,1,1,1,1,1,1,1,1,1};
        const int fseg1[] = {3,6,6,3,3,4,4,4,6,6};

        const int *seg, *vseg, *fseg;
        if (spin == XC_POLARIZED) {
                seg = seg1;
                vseg = vseg1;
                fseg = fseg1;
        } else {
                seg = seg0;
                vseg = vseg0;
                fseg = fseg0;
        }

        const int nv = 4;
        const int nf = 10;
        double *v[nv], *f[nf];
        if (vxc != NULL) {
                v[0] = vxc;
                for (i = 0; i < nv-1; i++) {
                    v[i+1] = v[i] + np * vseg[i];
                }
        }
        if (fxc != NULL) {
                f[0] = fxc;
                for (i = 0; i < nf-1; i++) {
                    f[i+1] = f[i] + np * fseg[i];
                }
        }

#pragma omp parallel private(i)
{
        int nblk = omp_get_num_threads();
        if (np < nblk) {nblk = 1;}
        int blk_size = np / nblk;

        int iblk;
        double *prho, *psigma, *plapl, *ptau, *pex;
        double *pv[nv], *pf[nf];
        for (i = 0; i < nv; i++) {pv[i] = NULL;}
        for (i = 0; i < nf; i++) {pf[i] = NULL;}
        #pragma omp for schedule(static)
        for (iblk = 0; iblk < nblk; iblk++) {
                prho = rho + iblk * blk_size * seg[0];
                psigma = sigma + iblk * blk_size * seg[1];
                plapl = lapl + iblk * blk_size * seg[2];
                ptau = tau + iblk * blk_size * seg[3];
                pex = ex + iblk * blk_size;
                if (vxc != NULL) {
                        for (i = 0; i < nv; i++) {
                                pv[i] = v[i] + iblk * blk_size * vseg[i];
                        } 
                }
                if (fxc != NULL) {
                        for (i = 0; i < nf; i++) {
                                pf[i] = f[i] + iblk * blk_size * fseg[i];
                        }
                }
#if XC_MAJOR_VERSION >=5
                xc_mgga_exc_vxc_fxc(func_x, blk_size, prho, psigma, plapl, ptau, pex,
                        pv[0], pv[1], pv[2], pv[3],
                        pf[0], pf[1], pf[2], pf[3], pf[4], pf[5],
                        pf[6], pf[7], pf[8], pf[9]);
#else
                xc_mgga(func_x, blk_size, prho, psigma, plapl, ptau, pex,
                        pv[0], pv[1], pv[2], pv[3],
                        pf[0], pf[1], pf[2], pf[3], pf[4], pf[5],
                        pf[6], pf[7], pf[8], pf[9]);
#endif
        }

#pragma omp single
{
        int np_res = np - nblk * blk_size;
        if (np_res > 0) {
                prho = rho + nblk * blk_size * seg[0];
                psigma = sigma + nblk * blk_size * seg[1];
                plapl = lapl + nblk * blk_size * seg[2];
                ptau = tau + nblk * blk_size * seg[3];
                pex = ex + nblk * blk_size;
                if (vxc != NULL) {
                        for (i = 0; i < nv; i++) {
                                pv[i] = v[i] + nblk * blk_size * vseg[i];
                        }
                }
                if (fxc != NULL) {
                        for (i = 0; i < nf; i++) {
                                pf[i] = f[i] + nblk * blk_size * fseg[i];
                        }
                }
#if XC_MAJOR_VERSION >=5
                xc_mgga_exc_vxc_fxc(func_x, np_res, prho, psigma, plapl, ptau, pex,
                        pv[0], pv[1], pv[2], pv[3],
                        pf[0], pf[1], pf[2], pf[3], pf[4], pf[5],
                        pf[6], pf[7], pf[8], pf[9]);
#else
                xc_mgga(func_x, np_res, prho, psigma, plapl, ptau, pex,
                        pv[0], pv[1], pv[2], pv[3],
                        pf[0], pf[1], pf[2], pf[3], pf[4], pf[5],
                        pf[6], pf[7], pf[8], pf[9]);
#endif
        }
} // omp single
} // omp parallel

        if (spin == XC_POLARIZED) {
                free(rho);
                free(lapl);
                free(tau);
        }
        free(sigma);
}


/* Extracted from comments of libxc:gga.c

    sigma_st          = grad rho_s . grad rho_t
    zk                = energy density per unit particle

    vrho_s            = d n*zk / d rho_s
    vsigma_st         = d n*zk / d sigma_st

    v2rho2_st         = d^2 n*zk / d rho_s d rho_t
    v2rhosigma_svx    = d^2 n*zk / d rho_s d sigma_tv
    v2sigma2_stvx     = d^2 n*zk / d sigma_st d sigma_vx

    v3rho3_stv        = d^3 n*zk / d rho_s d rho_t d rho_v
    v3rho2sigma_stvx  = d^3 n*zk / d rho_s d rho_t d sigma_vx
    v3rhosigma2_svxyz = d^3 n*zk / d rho_s d sigma_vx d sigma_yz
    v3sigma3_stvxyz   = d^3 n*zk / d sigma_st d sigma_vx d sigma_yz

 if nspin == 2
    rho(2)          = (u, d)
    sigma(3)        = (uu, ud, dd)

 * vxc(N*5):
    vrho(2)         = (u, d)
    vsigma(3)       = (uu, ud, dd)

 * fxc(N*45):
    v2rho2(3)       = (u_u, u_d, d_d)
    v2rhosigma(6)   = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
    v2sigma2(6)     = (uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)
    v2lapl2(3)
    vtau2(3)
    v2rholapl(4)
    v2rhotau(4)
    v2lapltau(4)
    v2sigmalapl(6)
    v2sigmatau(6)

 * kxc(N*35):
    v3rho3(4)       = (u_u_u, u_u_d, u_d_d, d_d_d)
    v3rho2sigma(9)  = (u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, d_d_ud, d_d_dd)
    v3rhosigma2(12) = (u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd)
    v3sigma(10)     = (uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, ud_ud_dd, ud_dd_dd, dd_dd_dd)

 */
/*
 * rho_u/rho_d = (den,grad_x,grad_y,grad_z,laplacian,tau)
 * In spin restricted case (spin == 1), rho_u is assumed to be the
 * spin-free quantities, rho_d is not used.
 */
static void _eval_xc(xc_func_type *func_x, int spin, int np,
                     double *rho_u, double *rho_d,
                     double *ex, double *vxc, double *fxc, double *kxc)
{
        switch (func_x->info->family) {
        case XC_FAMILY_LDA:
#ifdef XC_FAMILY_HYB_LDA
        case XC_FAMILY_HYB_LDA:
#endif
                // ex is the energy density
                // NOTE libxc library added ex/ec into vrho/vcrho
                // vrho = rho d ex/d rho + ex, see work_lda.c:L73
                _eval_xc_lda(func_x, spin, np, rho_u, rho_d, ex, vxc, fxc, kxc);
                break;
        case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
        case XC_FAMILY_HYB_GGA:
#endif
                _eval_xc_gga(func_x, spin, np, rho_u, rho_d, ex, vxc, fxc, kxc);
                break;
        case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
        case XC_FAMILY_HYB_MGGA:
#endif
                _eval_xc_mgga(func_x, spin, np, rho_u, rho_d, ex, vxc, fxc, kxc);
                break;
        default:
                fprintf(stderr, "functional %d '%s' is not implmented\n",
                        func_x->info->number, func_x->info->name);
                exit(1);
        }
}

int LIBXC_is_lda(int xc_id)
{
        xc_func_type func;
        int lda;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                exit(1);
        }
        switch(func.info->family)
        {
                case XC_FAMILY_LDA:
                        lda = 1;
                        break;
                default:
                        lda = 0;
        }

        xc_func_end(&func);
        return lda;
}

int LIBXC_is_gga(int xc_id)
{
        xc_func_type func;
        int gga;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                exit(1);
        }
        switch(func.info->family)
        {
                case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
                case XC_FAMILY_HYB_GGA:
#endif
                        gga = 1;
                        break;
                default:
                        gga = 0;
        }

        xc_func_end(&func);
        return gga;
}

int LIBXC_is_meta_gga(int xc_id)
{
        xc_func_type func;
        int mgga;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                exit(1);
        }
        switch(func.info->family)
        {
                case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
                case XC_FAMILY_HYB_MGGA:
#endif
                        mgga = 1;
                        break;
                default:
                        mgga = 0;
        }

        xc_func_end(&func);
        return mgga;
}

int LIBXC_needs_laplacian(int xc_id)
{
        xc_func_type func;
        int lapl;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                exit(1);
        }
        lapl = func.info->flags & XC_FLAGS_NEEDS_LAPLACIAN ? 1 : 0;
        xc_func_end(&func);
        return lapl;
}

int LIBXC_is_hybrid(int xc_id)
{
        xc_func_type func;
        int hyb;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                exit(1);
        }

#if XC_MAJOR_VERSION < 6
        switch(func.info->family)
        {
#ifdef XC_FAMILY_HYB_LDA
                case XC_FAMILY_HYB_LDA:
#endif
                case XC_FAMILY_HYB_GGA:
                case XC_FAMILY_HYB_MGGA:
                        hyb = 1;
                        break;
                default:
                        hyb = 0;
        }
#else
        hyb = (xc_hyb_type(&func) == XC_HYB_HYBRID);
#endif

        xc_func_end(&func);
        return hyb;
}

double LIBXC_hybrid_coeff(int xc_id)
{
        xc_func_type func;
        double factor;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                exit(1);
        }

#if XC_MAJOR_VERSION < 6
        switch(func.info->family)
        {
#ifdef XC_FAMILY_HYB_LDA
                case XC_FAMILY_HYB_LDA:
#endif
                case XC_FAMILY_HYB_GGA:
                case XC_FAMILY_HYB_MGGA:
                        factor = xc_hyb_exx_coef(&func);
                        break;
                default:
                        factor = 0;
        }

#else
        if(xc_hyb_type(&func) == XC_HYB_HYBRID)
          factor = xc_hyb_exx_coef(&func);
        else
          factor = 0.0;
#endif
        
        xc_func_end(&func);
        return factor;
}

void LIBXC_nlc_coeff(int xc_id, double *nlc_pars) {

        xc_func_type func;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                exit(1);
        }
        XC(nlc_coef)(&func, &nlc_pars[0], &nlc_pars[1]);
        xc_func_end(&func);
}

void LIBXC_rsh_coeff(int xc_id, double *rsh_pars) {

        xc_func_type func;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                exit(1);
        }
        rsh_pars[0] = 0.0;
        rsh_pars[1] = 0.0;
        rsh_pars[2] = 0.0;

#if XC_MAJOR_VERSION < 6
        XC(hyb_cam_coef)(&func, &rsh_pars[0], &rsh_pars[1], &rsh_pars[2]);
#else
        switch(xc_hyb_type(&func)) {
        case(XC_HYB_HYBRID):
        case(XC_HYB_CAM):
          XC(hyb_cam_coef)(&func, &rsh_pars[0], &rsh_pars[1], &rsh_pars[2]);
        }
#endif
        xc_func_end(&func);
}

int LIBXC_is_cam_rsh(int xc_id) {
        xc_func_type func;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                exit(1);
        }
#if XC_MAJOR_VERSION < 6
        int is_cam = func.info->flags & XC_FLAGS_HYB_CAM;
#else
        int is_cam = (xc_hyb_type(&func) == XC_HYB_CAM);
#endif
        xc_func_end(&func);
        return is_cam;
}

/*
 * XC_FAMILY_LDA           1
 * XC_FAMILY_GGA           2
 * XC_FAMILY_MGGA          4
 * XC_FAMILY_LCA           8
 * XC_FAMILY_OEP          16
 * XC_FAMILY_HYB_GGA      32
 * XC_FAMILY_HYB_MGGA     64
 * XC_FAMILY_HYB_LDA     128
 */
int LIBXC_xc_type(int fn_id)
{
        xc_func_type func;
        if (xc_func_init(&func, fn_id, 1) != 0) {
                fprintf(stderr, "XC functional %d not found\n", fn_id);
                exit(1);
        }
        int type = func.info->family;
        xc_func_end(&func);
        return type;
}

static int xc_output_length(int nvar, int deriv)
{
        int i;
        int len = 1.;
        for (i = 1; i <= nvar; i++) {
                len *= deriv + i;
                len /= i;
        }
        return len;
}

// return value 0 means no functional needs to be evaluated.
int LIBXC_input_length(int nfn, int *fn_id, double *fac, int spin)
{
        int i;
        int nvar = 0;
        xc_func_type func;
        for (i = 0; i < nfn; i++) {
                if (xc_func_init(&func, fn_id[i], spin) != 0) {
                        fprintf(stderr, "XC functional %d not found\n",
                                fn_id[i]);
                        exit(1);
                }
                if (spin == XC_POLARIZED) {
                        switch (func.info->family) {
                        case XC_FAMILY_LDA:
#ifdef XC_FAMILY_HYB_LDA
                        case XC_FAMILY_HYB_LDA:
#endif
                                nvar = MAX(nvar, 2);
                                break;
                        case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
                        case XC_FAMILY_HYB_GGA:
#endif
                                nvar = MAX(nvar, 5);
                                break;
                        case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
                        case XC_FAMILY_HYB_MGGA:
#endif
                                nvar = MAX(nvar, 9);
                        }
                } else {
                        switch (func.info->family) {
                        case XC_FAMILY_LDA:
#ifdef XC_FAMILY_HYB_LDA
                        case XC_FAMILY_HYB_LDA:
#endif
                                nvar = MAX(nvar, 1);
                                break;
                        case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
                        case XC_FAMILY_HYB_GGA:
#endif
                                nvar = MAX(nvar, 2);
                                break;
                        case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
                        case XC_FAMILY_HYB_MGGA:
#endif
                                nvar = MAX(nvar, 4);
                        }
                }
                xc_func_end(&func);
        }
        return nvar;
}

static void axpy(double *dst, double *src, double fac,
                 int np, int ndst, int nsrc)
{
        int i, j;
        for (j = 0; j < nsrc; j++) {
                #pragma omp parallel for schedule(static)
                for (i = 0; i < np; i++) {
                        dst[j*np+i] += fac * src[i*nsrc+j];
                }
        }
}

static void merge_xc(double *dst, double *ebuf, double *vbuf,
                     double *fbuf, double *kbuf, double fac,
                     int np, int ndst, int nvar, int spin, int type)
{
        const int seg0 [] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        // LDA             |  |
        // GGA             |     |
        // MGGA            |           |
        const int vseg1[] = {2, 3, 2, 2};
        // LDA             |  |
        // GGA             |        |
        // MGGA            |                             |
        const int fseg1[] = {3, 6, 6, 3, 3, 4, 4, 4, 6, 6};
        // LDA             |  |
        // GGA             |           |
        const int kseg1[] = {4, 9,12,10};
        int vsegtot, fsegtot, ksegtot;
        const int *vseg, *fseg, *kseg;
        if (spin == XC_POLARIZED) {
                vseg = vseg1;
                fseg = fseg1;
                kseg = kseg1;
        } else {
                vseg = seg0;
                fseg = seg0;
                kseg = seg0;
        }

        switch (type) {
        case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
        case XC_FAMILY_HYB_GGA:
#endif
                vsegtot = 2;
                fsegtot = 3;
                ksegtot = 4;
                break;
        case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
        case XC_FAMILY_HYB_MGGA:
#endif
                vsegtot = 4;
                fsegtot = 10;
                ksegtot = 0;  // not supported
                break;
        default: //case XC_FAMILY_LDA:
                vsegtot = 1;
                fsegtot = 1;
                ksegtot = 1;
        }

        int i;
        size_t offset;
        axpy(dst, ebuf, fac, np, ndst, 1);

        if (vbuf != NULL) {
                offset = np;
                for (i = 0; i < vsegtot; i++) {
                        axpy(dst+offset, vbuf, fac, np, ndst, vseg[i]);
                        offset += np * vseg[i];
                        vbuf += np * vseg[i];
                }
        }

        if (fbuf != NULL) {
                offset = np * xc_output_length(nvar, 1);
                for (i = 0; i < fsegtot; i++) {
                        axpy(dst+offset, fbuf, fac, np, ndst, fseg[i]);
                        offset += np * fseg[i];
                        fbuf += np * fseg[i];
                }
        }

        if (kbuf != NULL) {
                offset = np * xc_output_length(nvar, 2);
                for (i = 0; i < ksegtot; i++) {
                        axpy(dst+offset, kbuf, fac, np, ndst, kseg[i]);
                        offset += np * kseg[i];
                        kbuf += np * kseg[i];
                }
        }
}

// omega is the range separation parameter mu in xcfun
void LIBXC_eval_xc(int nfn, int *fn_id, double *fac, double *omega,
                   int spin, int deriv, int np,
                   double *rho_u, double *rho_d, double *output)
{
        assert(deriv <= 3);
        int nvar = LIBXC_input_length(nfn, fn_id, fac, spin);
        if (nvar == 0) { // No functional needs to be evaluated.
                return;
        }

        int outlen = xc_output_length(nvar, deriv);
        // output buffer is zeroed in the Python caller
        //NPdset0(output, np*outlen);

        double *ebuf = malloc(sizeof(double) * np);
        double *vbuf = NULL;
        double *fbuf = NULL;
        double *kbuf = NULL;
        if (deriv > 0) {
                vbuf = malloc(sizeof(double) * np*9);
        }
        if (deriv > 1) {
                fbuf = malloc(sizeof(double) * np*48);
        }
        if (deriv > 2) {  // *220 if mgga kxc available
                kbuf = malloc(sizeof(double) * np*35);
        }

        int i, j;
        xc_func_type func;
        for (i = 0; i < nfn; i++) {
                if (xc_func_init(&func, fn_id[i], spin) != 0) {
                        fprintf(stderr, "XC functional %d not found\n",
                                fn_id[i]);
                        exit(1);
                }

                // set the range-separated parameter
                if (omega[i] != 0) {
                        // skip if func is not a RSH functional
#if XC_MAJOR_VERSION < 6
                        if (func.cam_omega != 0) {
                                func.cam_omega = omega[i];
                        }
#else
                        if (func.hyb_omega[0] != 0) {
                                func.hyb_omega[0] = omega[i];
                        }
#endif
                        // Recursively set the sub-functionals if they are RSH
                        // functionals
                        for (j = 0; j < func.n_func_aux; j++) {
#if XC_MAJOR_VERSION < 6
                                if (func.func_aux[j]->cam_omega != 0) {
                                        func.func_aux[j]->cam_omega = omega[i];
                                }
#else
                                if (func.func_aux[j]->hyb_omega[0] != 0) {
                                        func.func_aux[j]->hyb_omega[0] = omega[i];
                                }
#endif
                        }
                }

                // alpha and beta are hardcoded in many functionals in the libxc
                // code, e.g. the coefficients of B88 (=1-alpha) and
                // ITYH (=-beta) in cam-b3lyp.  Overwriting func->cam_alpha and
                // func->cam_beta does not update the coefficients accordingly.
                //func->cam_alpha = alpha;
                //func->cam_beta  = beta;
                // However, the parameters can be set with the libxc function
                //void xc_func_set_ext_params_name(xc_func_type *p, const char *name, double par);
                // since libxc 5.1.0
#if defined XC_SET_RELATIVITY
                xc_lda_x_set_params(&func, relativity);
#endif
                _eval_xc(&func, spin, np, rho_u, rho_d, ebuf, vbuf, fbuf, kbuf);
                merge_xc(output, ebuf, vbuf, fbuf, kbuf, fac[i],
                         np, outlen, nvar, spin, func.info->family);
                xc_func_end(&func);
        }

        free(ebuf);
        if (deriv > 0) {
                free(vbuf);
        }
        if (deriv > 1) {
                free(fbuf);
        }
        if (deriv > 2) {
                free(kbuf);
        }
}

int LIBXC_max_deriv_order(int xc_id)
{
        xc_func_type func;
        int ord;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                exit(1);
        }

        if (func.info->flags & XC_FLAGS_HAVE_LXC) {
                ord = 4;
        } else if(func.info->flags & XC_FLAGS_HAVE_KXC) {
                ord = 3;
        } else if(func.info->flags & XC_FLAGS_HAVE_FXC) {
                ord = 2;
        } else if(func.info->flags & XC_FLAGS_HAVE_VXC) {
                ord = 1;
        } else if(func.info->flags & XC_FLAGS_HAVE_EXC) {
                ord = 0;
        } else {
                ord = -1;
        }

        xc_func_end(&func);
        return ord;
}

int LIBXC_number_of_functionals()
{
  return xc_number_of_functionals();
}

void LIBXC_functional_numbers(int *list)
{
  return xc_available_functional_numbers(list);
}

char * LIBXC_functional_name(int ifunc)
{
  return xc_functional_get_name(ifunc);
}

const char * LIBXC_version()
{
  return xc_version_string();
}

const char * LIBXC_reference()
{
  return xc_reference();
}

const char * LIBXC_reference_doi()
{
  return xc_reference_doi();
}

void LIBXC_xc_reference(int xc_id, const char **refs)
{
        xc_func_type func;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                exit(1);
        }

        int i;
        for (i = 0; i < XC_MAX_REFERENCES; i++) {
                if (func.info->refs[i] == NULL || func.info->refs[i]->ref == NULL) {
                        refs[i] = NULL;
                        break;
                }
                refs[i] = func.info->refs[i]->ref;
        }
}
