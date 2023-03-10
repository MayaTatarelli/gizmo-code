/* --------------------------------------------------------------------------------- */
/* ... explicit radiation diffusion/flux transport evaluation ...
 *
 * For SPH, this relys on the (noisy and zeroth-order inconsistent) SPH second-derivative
 *  operator. So a large kernel is especially useful to minimize the systematic errors.
 *  For MFM/MFV methods, the consistent finite-volume formulation is used.
 *  In either case, since we solve the conduction equations explicitly, a stronger timestep
 *  restriction is necessary (since the equations are parabolic); this is in timestep.c
 *
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
/* --------------------------------------------------------------------------------- */
double c_light = RT_SPEEDOFLIGHT_REDUCTION * (C/All.UnitVelocity_in_cm_per_s);
{
#if !defined(RT_EVOLVE_FLUX) /* this means we just solve the diffusion equation for the eddington tensor, done in the loop below */
    int k_freq;
    for(k_freq=0;k_freq<N_RT_FREQ_BINS;k_freq++)
    {
        Fluxes_E_gamma[k_freq] = 0;
        double kappa_ij = 0.5 * (local.RT_DiffusionCoeff[k_freq] + rt_diffusion_coefficient(j,k_freq)); // physical
        if((kappa_ij>0)&&(local.Mass>0)&&(P[j].Mass>0))
        {
            double scalar_i = local.E_gamma[k_freq] / V_i; // volumetric photon number density in this frequency bin (1/code volume) //
            double scalar_j = SphP[j].E_gamma_Pred[k_freq] / V_j;
            
            double d_scalar = (scalar_i - scalar_j); // units (1/code volume)
            double conduction_wt = kappa_ij * All.cf_a3inv/All.cf_atime;  // weight factor and conversion to physical units
            double cmag=0., grad_norm=0, grad_dot_x_ij=0.0;
            for(k=0;k<3;k++)
            {
                /* the flux is determined by the energy density gradient */
                double grad = 0.5 * (local.Gradients.E_gamma_ET[k_freq][k] + SphP[j].Gradients.E_gamma_ET[k_freq][k]); // (1/(code volume*code length))
                double grad_direct = d_scalar * kernel.dp[k] * rinv*rinv; // (1/(code volume*code length))
                grad_dot_x_ij += grad * kernel.dp[k];
                grad = MINMOD_G( grad , grad_direct );
#if defined(GALSF) || defined(COOLING) || defined(BLACKHOLES)
                double grad_direct_vs_abs_fac = 2.0;
#else
                double grad_direct_vs_abs_fac = 5.0;
#endif
                if(grad*grad_direct < 0) {if(fabs(grad_direct) > grad_direct_vs_abs_fac*fabs(grad)) {grad = 0.0;}}
                cmag += Face_Area_Vec[k] * grad;
                grad_norm += grad*grad;
            }
            double A_dot_grad_alignment = cmag*cmag / (Face_Area_Norm*Face_Area_Norm * grad_norm); // dimensionless
            cmag *= -conduction_wt; // multiplies through the coefficient to get actual flux (physical) //

            /* here we add the HLL-like correction term. this greatly reduces noise and improves the stability of the diffusion.
            	however it comes at the cost of (significant) additional numerical diffusion */
            double v_eff_light = DMIN(c_light , kappa_ij / (Get_Particle_Size(j)*All.cf_atime)); // physical
            double c_hll = 0.5*fabs(face_vel_i-face_vel_j) + v_eff_light;
            double q = 0.5 * c_hll * kernel.r * All.cf_atime / fabs(1.e-37 + kappa_ij); q = (0.2 + q) / (0.2 + q + q*q); // physical
            double d_scalar_tmp = d_scalar - grad_dot_x_ij; // (1/code volume)
            double d_scalar_hll = MINMOD(d_scalar , d_scalar_tmp) * All.cf_a3inv; // physical
            double hll_tmp = -A_dot_grad_alignment * q * Face_Area_Norm * c_hll * d_scalar_hll; // physical
            
            /* add asymptotic-preserving correction so that numerical flux doesn't dominate in optically thick limit */
            double tau_c_j = Get_Particle_Size(j)*All.cf_atime * SphP[j].Kappa_RT[k_freq]*SphP[j].Density*All.cf_a3inv; // = L_particle / (lambda_mean_free_path) = L*kappa*rho (physical) //
            double hll_corr = 1./(1. + 1.5*DMAX(tau_c_i[k_freq],tau_c_j));
            hll_tmp *= hll_corr;
            
            double thold_hll = 2.0*fabs(cmag);
            if(fabs(hll_tmp)>thold_hll) {hll_tmp*=thold_hll/fabs(hll_tmp);}
            double cmag_corr = cmag + hll_tmp;
            cmag = MINMOD(HLL_DIFFUSION_COMPROMISE_FACTOR*cmag, cmag_corr);
            /* flux-limiter to ensure flow is always down the local gradient */
            double f_direct = -conduction_wt * (1./9.) * Face_Area_Norm*d_scalar*rinv; // physical
            double check_for_stability_sign = f_direct*cmag;
            if((check_for_stability_sign < 0) && (fabs(f_direct) > HLL_DIFFUSION_OVERSHOOT_FACTOR*fabs(cmag))) {cmag = 0;}

            // prevent super-luminal local fluxes //
            double R_flux = fabs(cmag) / (3. * Face_Area_Norm * c_light * (fabs(d_scalar)*All.cf_a3inv) + 1.e-37); // physical
            R_flux = (1. + 12.*R_flux) / (1. + 12.*R_flux*(1.+R_flux)); // 12 arbitrary but >>1 gives good behavior here //
#ifndef FREEZE_HYDRO
            cmag *= R_flux;
#endif
            cmag *= dt_hydrostep; // all in physical units //
            if(fabs(cmag) > 0)
            {
                // enforce a flux limiter for stability (to prevent overshoot) //
                double thold_hll = 0.25 * DMIN(fabs(scalar_i*V_i-scalar_j*V_j),DMAX(fabs(scalar_i*V_i),fabs(scalar_j*V_j))); // physical
                if(check_for_stability_sign<0) {thold_hll *= 1.e-2;}
                if(fabs(cmag)>thold_hll) {cmag *= thold_hll/fabs(cmag);}
                cmag /= dt_hydrostep;
                Fluxes_E_gamma[k_freq] += cmag;
#ifdef RT_INFRARED
                // define advected radiation temperature based on direction of net radiation flow //
                if(k_freq==RT_FREQ_BIN_INFRARED) {if(cmag > 0) {Fluxes_E_gamma_T_weighted_IR = cmag/(MIN_REAL_NUMBER+SphP[j].Radiation_Temperature);} else {Fluxes_E_gamma_T_weighted_IR = cmag/(MIN_REAL_NUMBER+local.Radiation_Temperature);}}
#endif
            } // if(conduction_wt > 0)
            
        } // close check that kappa and particle masses are positive
    }


#else /* RT_EVOLVE_FLUX is ON, so we don't solve a diffusion equation, but a system of two standard advection-like equations */


    int k_freq;
    double c_hll = 0.5*fabs(face_vel_i-face_vel_j) + c_light; // physical units
    double V_i_phys = V_i / All.cf_a3inv;
    double V_j_phys = V_j / All.cf_a3inv;
    double sthreeinv = 1./sqrt(3.);
    for(k_freq=0;k_freq<N_RT_FREQ_BINS;k_freq++)
    {
        Fluxes_E_gamma[k_freq] = 0;
        Fluxes_Flux[k_freq][0]=Fluxes_Flux[k_freq][1]=Fluxes_Flux[k_freq][2]=0;
        double scalar_i = local.E_gamma[k_freq] / V_i_phys; // volumetric photon number density in this frequency bin (E_phys/L_phys^3)//
        double scalar_j = SphP[j].E_gamma_Pred[k_freq] / V_j_phys;
        if((scalar_i>0)&&(scalar_j>0)&&(local.Mass>0)&&(P[j].Mass>0)&&(dt_hydrostep>0)&&(Face_Area_Norm>0))
        {
            double d_scalar = scalar_i - scalar_j;
            double face_dot_flux=0., cmag=0., cmag_flux[3]={0}, grad_norm=0, flux_i[3]={0}, flux_j[3]={0}, thold_hll;
            double kappa_ij = 0.5 * (local.RT_DiffusionCoeff[k_freq] + rt_diffusion_coefficient(j,k_freq)); // physical
            
            /* calculate the eigenvalues for the HLLE flux-weighting */
            for(k=0;k<3;k++)
            {
                flux_i[k] = local.Flux[k_freq][k]/V_i_phys; flux_j[k] = SphP[j].Flux_Pred[k_freq][k]/V_j_phys; // units (E_phys/[t_phys*L_phys^2])
                double grad = 0.5*(flux_i[k] + flux_j[k]);
                grad_norm += grad*grad;
                face_dot_flux += Face_Area_Vec[k] * grad; /* remember, our 'flux' variable is a volume-integral */
            }
            grad_norm = sqrt(grad_norm) + MIN_REAL_NUMBER;
            double reduced_flux = grad_norm / ((C/All.UnitVelocity_in_cm_per_s) * 0.5*(scalar_i+scalar_j)); // |F|/(c*E): ratio of flux to optically thin limit
            if(reduced_flux > 1) {reduced_flux=1;} else {if(reduced_flux < 0) {reduced_flux=0;}}
            double cos_theta_face_flux = face_dot_flux / (Face_Area_Norm * grad_norm); // angle between flux and face vector normal
            if(cos_theta_face_flux < -1) {cos_theta_face_flux=-1;} else {if(cos_theta_face_flux > 1) {cos_theta_face_flux=1;}}
            double lam_m, lam_p, wt, y_f=1.-reduced_flux, y_f_h=sqrt(y_f), y_f_h2=sqrt(y_f_h), cth=cos_theta_face_flux/2.;
            wt = (1. + cos_theta_face_flux)*(1. + cos_theta_face_flux) / 4.;
            lam_m = sthreeinv - reduced_flux*(+cth + (1.-(y_f_h2+wt*y_f_h)/(1.+wt))*(+cth+sthreeinv));
            wt = (1. - cos_theta_face_flux)*(1. - cos_theta_face_flux) / 4.;
            lam_p = sthreeinv - reduced_flux*(-cth + (1.-(y_f_h2+wt*y_f_h)/(1.+wt))*(-cth+sthreeinv));
            
            if(lam_p < 0) {lam_p=0;} else {if(lam_p>1) {lam_p=1;}}
            if(lam_m < 0) {lam_m=0;} else {if(lam_m>1) {lam_m=1;}}
            double hlle_wtfac_f, hlle_wtfac_u, eps_wtfac_f = 1.0e-10; // minimum weight
            if((lam_m==0)&&(lam_p==0)) {hlle_wtfac_f=0.5;} else {hlle_wtfac_f=lam_p/(lam_p+lam_m);}
            if(hlle_wtfac_f < eps_wtfac_f) {hlle_wtfac_f=eps_wtfac_f;} else {if(hlle_wtfac_f > 1.-eps_wtfac_f) {hlle_wtfac_f=1.-eps_wtfac_f;}}
            hlle_wtfac_u = hlle_wtfac_f * (1.-hlle_wtfac_f) * (lam_p + lam_m); // weight for addition of diffusion term
             
            for(k=0;k<3;k++)
            {
                /* the flux is already known (its explicitly evolved, rather than determined by the gradient of the energy density */
                cmag += Face_Area_Vec[k] * (hlle_wtfac_f*flux_i[k] + (1.-hlle_wtfac_f)*flux_j[k]); /* remember, our 'flux' variable is a volume-integral [all physical units here] */
                int k_xyz=k, k_et_al, k_et_loop[3];
                if(k_xyz==0) {k_et_loop[0]=0; k_et_loop[1]=3; k_et_loop[2]=5;}
                if(k_xyz==1) {k_et_loop[0]=3; k_et_loop[1]=1; k_et_loop[2]=4;}
                if(k_xyz==2) {k_et_loop[0]=5; k_et_loop[1]=4; k_et_loop[2]=2;}
                for(k_et_al=0;k_et_al<3;k_et_al++) {
                    cmag_flux[k_xyz] += c_light*c_light * Face_Area_Vec[k_et_al] * (hlle_wtfac_f*scalar_i*local.ET[k_freq][k_et_loop[k_et_al]] + (1.-hlle_wtfac_f)*scalar_j*SphP[j].ET[k_freq][k_et_loop[k_et_al]]);} // [all physical units]
            }
            
            /* add asymptotic-preserving correction so that numerical flux doesn't unphysically dominate in optically thick limit */
            double v_eff_light = DMIN(c_light , kappa_ij / (Get_Particle_Size(j)*All.cf_atime)); // physical
            c_hll = 0.5*fabs(face_vel_i-face_vel_j) + DMAX(1.,hlle_wtfac_u) * v_eff_light; // physical
            double tau_c_j = Get_Particle_Size(j)*All.cf_atime * SphP[j].Kappa_RT[k_freq]*(SphP[j].Density*All.cf_a3inv); // = L_particle / (lambda_mean_free_path) = L*kappa*rho [physical units] //
            double hll_corr = 1./(1. + 1.5*DMAX(tau_c_i[k_freq],tau_c_j));
            /* q below is a limiter to try and make sure the diffusion speed given by the hll flux doesn't exceed the diffusion speed in the diffusion limit */
            double q = 0.5 * c_hll * (kernel.r*All.cf_atime) / fabs(MIN_REAL_NUMBER + kappa_ij); q = (0.2 + q) / (0.2 + q + q*q); // physical
            double renormerFAC = DMIN(1.,fabs(cos_theta_face_flux*cos_theta_face_flux * q * hll_corr));            
            
            
            /* flux-limiter to ensure flow is always down the local gradient [no 'uphill' flow] */
            double f_direct = -Face_Area_Norm * c_hll * d_scalar * renormerFAC; // simple HLL term for frame moving at 1/2 inter-particle velocity: here not limited [physical units] //
            double sign_c0 = f_direct*cmag;
            if((sign_c0 < 0) && (fabs(f_direct) > fabs(cmag))) {cmag = 0;}
            if(cmag != 0)
            {
                if(f_direct != 0)
                {
                    thold_hll = (0.5*hlle_wtfac_u) * fabs(cmag); // add hll term but flux-limited //
                    if(fabs(f_direct) > thold_hll) {f_direct *= thold_hll/fabs(f_direct);}
                    cmag += f_direct;
                }
                // enforce a flux limiter for stability (to prevent overshoot) //
                cmag *= dt_hydrostep; // all in physical units //
                double sVi = scalar_i*V_i_phys, sVj = scalar_j*V_j_phys; // physical units //
                thold_hll = 0.25 * DMIN(fabs(sVi-sVj), DMAX(fabs(sVi), fabs(sVj)));
                if(sign_c0 < 0) {thold_hll *= 1.e-2;} // if opposing signs, restrict this term //
                if(fabs(cmag)>thold_hll) {cmag *= thold_hll/fabs(cmag);}
                cmag /= dt_hydrostep;
                Fluxes_E_gamma[k_freq] += cmag; // returned in physical units //
#ifdef RT_INFRARED
                // define advected radiation temperature based on direction of net radiation flow //
                if(k_freq==RT_FREQ_BIN_INFRARED) {if(cmag > 0) {Fluxes_E_gamma_T_weighted_IR = cmag/(MIN_REAL_NUMBER+SphP[j].Radiation_Temperature);} else {Fluxes_E_gamma_T_weighted_IR = cmag/(MIN_REAL_NUMBER+local.Radiation_Temperature);}}
#endif
            } // cmag != 0
            
            /* alright, now we calculate and limit the HLL diffusive fluxes for the radiative fluxes */
            double hll_mult_dmin = 1;
            for(k=0;k<3;k++)
            {
                // going to 0.5 instead of 1 here strongly increases diffusion: still get squared HII region, but very bad shadowing
                double f_direct = -Face_Area_Norm * c_hll * (flux_i[k]-flux_j[k]); // * renormerFAC; [all physical units]
                if(f_direct != 0)
                {
                    thold_hll = 1.0 * fabs(cmag_flux[k]) / fabs(f_direct); // coefficient of 0.5-1: 0.5=longer-lasting shadows, more 'memory' effect/shape distortion of HII regions; 1=fill in shadows faster, less HII distortion
//                    thold_hll = 0.5 * fabs(cmag_flux[k]) / fabs(f_direct); // coefficient of 0.5-1: 0.5=longer-lasting shadows, more 'memory' effect/shape distortion of HII regions; 1=fill in shadows faster, less HII distortion
                    if(thold_hll < hll_mult_dmin) {hll_mult_dmin = thold_hll;}
                }
            }
            for(k=0;k<3;k++)
            {
                double f_direct = -Face_Area_Norm * c_hll * (flux_i[k] - flux_j[k]); // [physical units]
                double sign_agreement = f_direct * cmag_flux[k];
                if((sign_agreement < 0) && (fabs(f_direct) > fabs(cmag_flux[k]))) {cmag_flux[k] = 0;}
                if(cmag_flux[k] != 0)
                {
                    cmag_flux[k] += hll_mult_dmin * f_direct; // add diffusive flux //
                    /* flux-limiter to prevent overshoot */
                    cmag_flux[k] *= dt_hydrostep;
                    thold_hll = DMIN( DMAX( DMIN(fabs(local.Flux[k_freq][k]), fabs(SphP[j].Flux_Pred[k_freq][k])) , fabs(local.Flux[k_freq][k]-SphP[j].Flux_Pred[k_freq][k]) ) , DMAX(fabs(local.Flux[k_freq][k]), fabs(SphP[j].Flux_Pred[k_freq][k])) );
                    double fii=V_i_phys*scalar_i*c_light, fjj=V_j_phys*scalar_j*c_light; // physical units //
                    double tij = DMIN( DMAX( DMIN(fabs(fii),fabs(fjj)) , fabs(fii-fjj) ) , DMAX(fabs(fii),fabs(fjj)) );
                    thold_hll = 0.25 * DMAX( thold_hll , 0.5*tij );
                    if(fabs(cmag_flux[k])>thold_hll) {cmag_flux[k] *= thold_hll/fabs(cmag_flux[k]);}
                    Fluxes_Flux[k_freq][k] += cmag_flux[k] / dt_hydrostep; // all returned in physical units
                }
            }
        } // close check that energy and masses are positive
    }

#endif
    
    // assign the actual fluxes //
    for(k=0;k<N_RT_FREQ_BINS;k++) {out.Dt_E_gamma[k] += Fluxes_E_gamma[k];}
    if(j_is_active_for_fluxes) {for(k=0;k<N_RT_FREQ_BINS;k++) {SphP[j].Dt_E_gamma[k] -= Fluxes_E_gamma[k];}}
#if defined(RT_INFRARED)
    out.Dt_E_gamma_T_weighted_IR += Fluxes_E_gamma_T_weighted_IR;
    if(j_is_active_for_fluxes) {SphP[j].Dt_E_gamma_T_weighted_IR -= Fluxes_E_gamma_T_weighted_IR;}
#endif
#ifdef RT_EVOLVE_FLUX
    for(k=0;k<N_RT_FREQ_BINS;k++) {int k_dir; for(k_dir=0;k_dir<3;k_dir++) {out.Dt_Flux[k][k_dir] += Fluxes_Flux[k][k_dir];}}
    if(j_is_active_for_fluxes) {for(k=0;k<N_RT_FREQ_BINS;k++) {int k_dir; for(k_dir=0;k_dir<3;k_dir++) {SphP[j].Dt_Flux[k][k_dir] -= Fluxes_Flux[k][k_dir];}}}
#endif
    
}
