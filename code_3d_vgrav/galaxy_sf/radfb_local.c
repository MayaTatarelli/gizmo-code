#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "../allvars.h"
#include "../proto.h"
#include "../kernel.h"

/* this file handles the FIRE short-range radiation-pressure and
    photo-ionization terms. written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#if defined(GALSF_FB_FIRE_RT_LOCALRP) /* first the radiation pressure coupled in the immediate vicinity of the star */
void radiation_pressure_winds_consolidated(void)
{
    MyDouble *pos; int N_MAX_KERNEL,N_MIN_KERNEL,MAXITER_FB,NITER,startnode,dummy,numngb_inbox,i,j,k,n;
    double dx,dy,dz,r2,u,h,hinv,hinv3,wk,rho,wt_sum,p_random,p_cumulative,star_age,lm_ssp,dv_units,dE_over_c,unitmass_in_msun,prob,dt,v,vq=0,dv_imparted,dv_imparted_uv,norm,dir[3], total_n_wind,total_m_wind,total_mom_wind,total_prob_kick,avg_v_kick,momwt_avg_v_kick,avg_taufac;
    double totMPI_n_wind,totMPI_m_wind,totMPI_mom_wind,totMPI_prob_kick,totMPI_avg_v,totMPI_pwt_avg_v,totMPI_taufac, sigma_eff_0, RtauMax = 0, age_thold = 0.1;
    total_n_wind=total_m_wind=total_mom_wind=total_prob_kick=avg_v_kick=momwt_avg_v_kick=avg_taufac=0; totMPI_n_wind=totMPI_m_wind=totMPI_mom_wind=totMPI_prob_kick=totMPI_avg_v=totMPI_pwt_avg_v=totMPI_taufac=0; p_random=p_cumulative=0;
#ifdef SINGLE_STAR_FORMATION
    age_thold = 1.0e10;
#endif
    if(All.WindMomentumLoading<=0) return;
    Ngblist = (int *) mymalloc("Ngblist",NumPart * sizeof(int));
#ifndef IO_REDUCED_MODE
    if(ThisTask == 0) {printf("Beginning Local Radiation-Pressure Acceleration\n");} // if(ThisTask == 0)
#endif
    
    unitmass_in_msun=(All.UnitMass_in_g/All.HubbleParam)/SOLAR_MASS; sigma_eff_0 = 0.955 * All.UnitMass_in_g*All.HubbleParam/(All.UnitLength_in_cm*All.UnitLength_in_cm) / (All.cf_atime*All.cf_atime) * KAPPA_IR;
    double unitlength_in_kpc=All.UnitLength_in_cm/All.HubbleParam/3.086e21*All.cf_atime;
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if((P[i].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[i].Type == 2)||(P[i].Type==3))))
        {
            
#ifndef WAKEUP
            dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval / All.cf_hubble_a;
#else
            dt = P[i].dt_step * All.Timebase_interval / All.cf_hubble_a;
#endif
            star_age = evaluate_stellar_age_Gyr(P[i].StellarAge);
            if( (star_age < age_thold) && (P[i].Mass > 0) && (P[i].DensAroundStar > 0) )
            {
                RtauMax = P[i].Hsml*All.cf_atime * (2.0 * KAPPA_UV * P[i].Hsml*P[i].DensAroundStar/(All.cf_atime*All.cf_atime) * All.UnitDensity_in_cgs*All.HubbleParam*All.UnitLength_in_cm);
                RtauMax /= All.cf_atime; RtauMax += 5.*P[i].Hsml;
                double rmax0 = 10.0 / unitlength_in_kpc; if(RtauMax > rmax0) RtauMax = rmax0;
                rmax0 = 1.0 / unitlength_in_kpc; if(RtauMax < rmax0) RtauMax = rmax0;
#ifndef GALSF_FB_FIRE_RT_CONTINUOUSRP
                /* if kicks are stochastic, we don't want to waste time doing a neighbor search every timestep; it can be much faster to pre-estimate the kick probabilities */
                double v_wind_threshold = 15.e5 / All.UnitVelocity_in_cm_per_s;
#ifdef SINGLE_STAR_FORMATION
                v_wind_threshold = 0.2e5 / All.UnitVelocity_in_cm_per_s;
#endif
                rho=P[i].DensAroundStar; h=P[i].Hsml;
                v = sqrt( All.G * (P[i].Mass + NORM_COEFF*rho*h*h*h) / (h*All.cf_atime) ); if(vq<v) v=vq;
                vq = 1.82*(65.748e5/All.UnitVelocity_in_cm_per_s) * pow(1.+rho*All.cf_a3inv*All.UnitDensity_in_cgs*All.HubbleParam*All.HubbleParam/(1.67e-24),-0.25);
                /* this corresponds to =G M_star/R_e for a 10^6 Msun cluster, scaling upwards from there; note that All.WindEnergyFraction will boost appropriately; should be at least sqrt(2), if want full velocities; in fact for Hernquist profile, the mass-weighted V_esc=1.82 times this */
                if(vq<v) {v=vq;}
                if(v<=v_wind_threshold) v=v_wind_threshold;
                lm_ssp = evaluate_light_to_mass_ratio(star_age, i);
                dE_over_c = lm_ssp * (4.0/2.0) * (P[i].Mass*All.UnitMass_in_g/All.HubbleParam); // L in CGS
                dE_over_c *= (dt*All.UnitTime_in_s/All.HubbleParam) / 2.9979e10; // dE/c in CGS
                dv_units = KAPPA_IR * dE_over_c / (4*M_PI * All.UnitLength_in_cm*All.UnitLength_in_cm*All.cf_atime*All.cf_atime);
                dv_units /= All.UnitVelocity_in_cm_per_s; // dv in code units per unit distance
                dv_units *= All.WindMomentumLoading; // rescale tau_ir component here
                dE_over_c /= (All.UnitMass_in_g/All.HubbleParam) * All.UnitVelocity_in_cm_per_s; // dv per unit mass
                total_prob_kick += dE_over_c; dv_imparted = dE_over_c/P[i].Mass; // estimate of summed dv_imparted from neighbors from L/c part
                dv_imparted += dv_units * (0.1+P[i].Metallicity[0]/All.SolarAbundances[0]) * (4.0*M_PI*rho/P[i].Mass*h); // sum over neighbor IR term
                prob = dv_imparted / v; prob *= 2000.; // need to include a buffer for errors in the estimates above
                p_random = get_random_number(P[i].ID+ThisTask+i+2); // master random number for use below
                p_cumulative = 0; // used below if the loop is executed
                if(p_random <= prob) // alright, its worth doing the loop!
#endif
                { // within loop
                    /* ok, now open the neighbor list for the star particle */
                    N_MIN_KERNEL=2;N_MAX_KERNEL=500;MAXITER_FB=5;NITER=0;rho=0;wt_sum=0; startnode=All.MaxPart;dummy=0;numngb_inbox=0;h=1.0*P[i].Hsml;pos=P[i].Pos;
                    if(h<=0) {h=All.SofteningTable[0];} else {if(h>RtauMax) {h=RtauMax;}}
                    do {
                        numngb_inbox = ngb_treefind_pairs_threads(pos, h, -1, &startnode, 0, &dummy, &dummy, &dummy, Ngblist);
                        if((numngb_inbox>=N_MIN_KERNEL)&&(numngb_inbox<=N_MAX_KERNEL))
                        {
                            hinv=1/h; hinv3=hinv*hinv*hinv; wt_sum=rho=0;
                            for(n=0; n<numngb_inbox; n++)
                            {
                                j = Ngblist[n];
                                if( (P[j].Mass>0) && (SphP[j].Density>0) )
                                {
                                    dx=P[j].Pos[0]-P[i].Pos[0]; dy=P[j].Pos[1]-P[i].Pos[1]; dz=P[j].Pos[2]-P[i].Pos[2]; r2 = dx*dx + dy*dy + dz*dz; r2 += MIN_REAL_NUMBER; // just a small number to prevent errors on near-overlaps
                                    double h_eff_i = Get_Particle_Size(i), h_eff_j = Get_Particle_Size(j); r2 += (h_eff_i/5.)*(h_eff_i/5.); // just a small number to prevent errors on near-overlaps
                                    u=sqrt(r2)*hinv; kernel_main(u,hinv3,1,&wk,&vq,-1); rho += (P[j].Mass*wk); wt_sum += h_eff_j*h_eff_j;// / r2;
                                } /* if( (P[j].Mass>0) && (SphP[j].Density>0) ) */
                            } /* for(n=0; n<numngb_inbox; n++) */
                            if (rho <= 0) {h*= 1.2123212335; startnode=All.MaxPart;} /* rho <= 0; no massive particles found, trigger a new loop */
                        }
                        else
                        {
                            startnode=All.MaxPart;
                            if(numngb_inbox<N_MIN_KERNEL)
                            {
                                if(numngb_inbox<=0) {h*=2.0;} else {if(NITER<=5) {h*=pow((float)numngb_inbox/(float)N_MIN_KERNEL,-0.3333);} else {h*=1.26;}} /* iterate until find appropriate > N_MIN # particles */
                            }
                            if(numngb_inbox>N_MAX_KERNEL) {if(NITER<=5) {h*=pow((float)numngb_inbox/(float)N_MAX_KERNEL,-0.3333);} else {h/=1.31;}} /* iterate until find appropriate < N_MAX # particles */
                        }
                        /* if h exceeds the maximum now, set it to that value, and set NITER to maximum to force end of iteration */
                        if(h>RtauMax) {h = RtauMax; if(NITER<MAXITER_FB-1) {NITER=MAXITER_FB-1;}}
                        NITER++;
                    } while( (startnode >= 0) && (NITER<=MAXITER_FB) );
                    
                    if (rho > 0)  /* found at least one massive neighbor, can proceed */
                    {
                        hinv=1/h; hinv3=hinv*hinv*hinv;
#ifndef GALSF_FB_FIRE_RT_CONTINUOUSRP
                        v = sqrt( All.G * (P[i].Mass + NORM_COEFF*rho*h*h*h) / (h*All.cf_atime) ); // re-calc v with our local rho estimate we just obtained //
                        if(vq<v) v=vq;
                        if(v<=v_wind_threshold) v=v_wind_threshold;
#else
                        lm_ssp = evaluate_light_to_mass_ratio(star_age, i);
                        dE_over_c = lm_ssp * (4.0/2.0) * (P[i].Mass*All.UnitMass_in_g/All.HubbleParam); // L in CGS
                        dE_over_c *= (dt*All.UnitTime_in_s/All.HubbleParam) / 2.9979e10; // dE/c in CGS
                        dv_units = KAPPA_IR * dE_over_c / (4*M_PI * All.UnitLength_in_cm*All.UnitLength_in_cm*All.cf_atime*All.cf_atime);
                        dv_units /= All.UnitVelocity_in_cm_per_s; // dv in code units per unit distance
                        dv_units *= All.WindMomentumLoading; // rescale tau_ir component here
                        dE_over_c /= (All.UnitMass_in_g/All.HubbleParam) * All.UnitVelocity_in_cm_per_s; // dv per unit mass
                        total_prob_kick += dE_over_c;
#endif
                        for(n=0; n<numngb_inbox; n++)
                        {
                            j = Ngblist[n];
                            if( (P[j].Mass>0) && (SphP[j].Density>0) )
                            {
                                dx=P[j].Pos[0]-P[i].Pos[0]; dy=P[j].Pos[1]-P[i].Pos[1]; dz=P[j].Pos[2]-P[i].Pos[2]; r2 = dx*dx + dy*dy + dz*dz; r2 += MIN_REAL_NUMBER; // just a small number to prevent errors on near-overlaps
                                double h_eff_i = Get_Particle_Size(i); r2 += (h_eff_i/5.)*(h_eff_i/5.); // just a small number to prevent errors on near-overlaps
                                /* velocity imparted by IR acceleration : = kappa*flux/c, flux scales as 1/r2 from source, kappa with metallicity */
                                dv_imparted = dv_units * (0.1 + P[j].Metallicity[0]/All.SolarAbundances[0]) / r2;
                                /* first loop -- share out the UV luminosity among the local neighbors, weighted by the gas kernel */
                                double h_eff_j = Get_Particle_Size(j); wk = h_eff_j*h_eff_j / wt_sum; dv_imparted_uv = wk * dE_over_c / P[j].Mass;
#ifdef GALSF_FB_FIRE_RT_CONTINUOUSRP
                                v = dv_imparted + dv_imparted_uv; prob = 1;
#else
                                prob = (dv_imparted+dv_imparted_uv) / v; if(prob>1) v *= prob;
                                if(n>0) p_random=get_random_number(P[j].ID+P[i].ID +ThisTask+ 3); //else p_random=0;
                                if(p_random < prob)
#endif
                                { /* open subloop with wind kick */
                                    if(v>5.0e8/All.UnitVelocity_in_cm_per_s) v=5.0e8/All.UnitVelocity_in_cm_per_s; /* limiter */
                                    /* collect numbers to output */
                                    total_n_wind += 1.0; total_mom_wind += P[j].Mass*v; avg_v_kick += v; momwt_avg_v_kick += P[j].Mass*v * sigma_eff_0*P[j].Mass/(h_eff_j*h_eff_j) * (0.01 + P[j].Metallicity[0]/All.SolarAbundances[0]);
                                    
                                    /* determine the direction of the kick */
#ifdef GALSF_FB_FIRE_RT_CONTINUOUSRP
                                    v = dv_imparted; // ir kick: directed along opacity gradient //
                                    for(k=0;k<3;k++) dir[k]=-P[j].GradRho[k]; // based on density gradient near star //
#else
                                    if(dv_imparted_uv > dv_imparted)
                                    {
                                        dir[0]=dx; dir[1]=dy; dir[2]=dz; // if kick is primarily from uv, then orient directly //
                                    } else {
                                        for(k=0;k<3;k++) dir[k]=-P[j].GradRho[k]; // otherwise, along opacity gradient //
                                    }
#endif
                                    norm=0; for(k=0; k<3; k++) {norm += dir[k]*dir[k];}
                                    if(norm>0) {norm=sqrt(norm); for(k=0;k<3;k++) dir[k] /= norm;} else {dir[0]=0; dir[1]=0; dir[2]=1; norm=1;}
                                    for(k=0; k<3; k++) {P[j].Vel[k] += v * All.cf_atime * dir[k]; SphP[j].VelPred[k] += v * All.cf_atime * dir[k];} /* apply the kick */
                                    
#if defined(GALSF_FB_FIRE_RT_CONTINUOUSRP) /* if we're not forcing the kick orientation, need to separately apply the UV kick */
                                    v = dv_imparted_uv; // uv kick: directed from star //
                                    dir[0]=dx; dir[1]=dy; dir[2]=dz; // based on density gradient near star //
                                    norm=0; for(k=0; k<3; k++) {norm += dir[k]*dir[k];}
                                    if(norm>0) {norm=sqrt(norm); for(k=0;k<3;k++) {dir[k] /= norm;}} else {dir[0]=0; dir[1]=0; dir[2]=1; norm=1;}
                                    for(k=0; k<3; k++) {P[j].Vel[k] += v * All.cf_atime * dir[k]; SphP[j].VelPred[k] += v * All.cf_atime * dir[k];} /* apply the kick */
#endif
                                } /* closes if(get_random_number(P[i].ID + 2) < prob) */
                            } /* if( (P[j].Mass>0) && (SphP[j].Density>0) ) */
                        } /* for(n=0; n<numngb_inbox; n++) */
                    } /* if (rho>0) */
                } // // within loop
            } // star age, mass check:: (star_age < 0.1) && (P[i].Mass > 0) && (P[i].DensAroundStar > 0)
        } // particle type check::  if((P[i].Type == 4)....
    } // main particle loop for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    myfree(Ngblist);
    
    MPI_Reduce(&total_n_wind, &totMPI_n_wind, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_mom_wind, &totMPI_mom_wind, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avg_v_kick, &totMPI_avg_v, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&momwt_avg_v_kick, &totMPI_pwt_avg_v, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_prob_kick, &totMPI_prob_kick, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0)
    {
        if(totMPI_prob_kick>0)
        {
#ifdef IO_REDUCED_MODE
            if(totMPI_n_wind>0)
#endif
            {
                if(totMPI_n_wind>0) {totMPI_avg_v /= totMPI_n_wind; totMPI_pwt_avg_v /= totMPI_mom_wind;}
                fprintf(FdMomWinds, "%lg %g %g %g %g %g \n", All.Time,totMPI_n_wind,totMPI_prob_kick,totMPI_mom_wind,totMPI_avg_v,totMPI_pwt_avg_v);
#ifndef IO_REDUCED_MODE
                printf("Momentum Wind Feedback: Time=%g Nkicked=%g (L/c)dt=%g Momkicks=%g V_avg=%g tau_j_mean=%g \n", All.Time,totMPI_n_wind,totMPI_prob_kick,totMPI_mom_wind,totMPI_avg_v,totMPI_pwt_avg_v); fflush(stdout);
#endif
            }
        }
#ifdef IO_REDUCED_MODE
        if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin)
#endif
        {fflush(FdMomWinds);}
    } // if(ThisTask==0)
} // end routine :: void radiation_pressure_winds_consolidated(void)

#endif /* closes defined(GALSF_FB_FIRE_RT_LOCALRP)  */


    
    
    
/* Routines for simple FIRE local photo-ionization heating feedback model. This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO. */
#if defined(GALSF_FB_FIRE_RT_HIIHEATING)
void HII_heating_singledomain(void)    /* this version of the HII routine only communicates with particles on the same processor */
{
#ifdef RT_CHEM_PHOTOION
    return; // the work here is done in the actual RT routines if this switch is enabled //
#endif
    if(All.HIIRegion_fLum_Coupled<=0) return;
    if(All.Time<=0) return;
    MyDouble *pos; MyFloat h_i, dt, rho;
    int startnode, numngb, j, n, i, NITER_HIIFB, MAX_N_ITERATIONS_HIIFB, jnearest,already_ionized,do_ionize,dummy;
    double totMPI_m_ionizing,totMPI_l_ionizing,totMPI_m_ionized,totMPI_m_ionizable,totMPI_avg_RHII,dx, dy, dz, h_i2, r2, r, u, u_to_temp_fac,mionizable,mionized,RHII,RHIIMAX,R_search,rnearest,stellum,uion,prob,rho_j,prandom,m_available,m_effective,RHII_initial,RHIImultiplier, total_l_ionizing,total_m_ionizing,total_m_ionizable,total_m_ionized,avg_RHII;
    total_l_ionizing=total_m_ionized=avg_RHII=total_m_ionizable=total_m_ionizing=0; totMPI_m_ionizing=totMPI_l_ionizing=totMPI_m_ionized=totMPI_m_ionizable=totMPI_avg_RHII=0;
    Ngblist = (int *) mymalloc("Ngblist",NumPart * sizeof(int));
    MAX_N_ITERATIONS_HIIFB = 5; NITER_HIIFB = 0;
    
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#ifdef BH_HII_HEATING
        if((P[i].Type==5)||(((P[i].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[i].Type == 2)||(P[i].Type==3))))))
#else
        if((P[i].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[i].Type == 2)||(P[i].Type==3))))
#endif
        {
#ifndef WAKEUP
            dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval / All.cf_hubble_a;
#else
            dt = P[i].dt_step * All.Timebase_interval / All.cf_hubble_a;
#endif
            if(dt<=0) continue; // don't keep going with this loop
            
            stellum = All.HIIRegion_fLum_Coupled * particle_ionizing_luminosity_in_cgs(i);
            if(stellum <= 0) continue;
            pos = P[i].Pos; rho = P[i].DensAroundStar; h_i = PPP[i].Hsml; total_m_ionizing += 1; total_l_ionizing += stellum;
            
            RHII = 4.67e-9*pow(stellum,0.333)*pow(rho*All.cf_a3inv*All.UnitDensity_in_cgs*All.HubbleParam*All.HubbleParam,-0.66667);
            RHII /= All.cf_atime*All.UnitLength_in_cm/All.HubbleParam;
            RHIIMAX=240.0*pow(stellum,0.5)/(All.cf_atime*All.UnitLength_in_cm/All.HubbleParam); // crude estimate of where flux falls below cosmic background
            if(RHIIMAX < h_i) RHIIMAX=h_i;
            if(RHIIMAX > 5.0*h_i) RHIIMAX=5.*h_i;
            mionizable=NORM_COEFF*rho*RHII*RHII*RHII;
            double M_ionizing_emitted = (3.05e10 * PROTONMASS) * stellum * (dt * All.UnitTime_in_s / All.HubbleParam) ; // number of ionizing photons times proton mass, gives max mass ionized
            mionizable = DMIN( mionizable , M_ionizing_emitted/(All.UnitMass_in_g/All.HubbleParam) );
            if(RHII>RHIIMAX) RHII=RHIIMAX;
            if(RHII<0.5*h_i) RHII=0.5*h_i;
            RHII_initial=RHII;
            
            prandom = get_random_number(P[i].ID + 7); // pre-calc the (eventually) needed random number
            // guesstimate if this is even close to being interesting for the particle masses of interest
            if(prandom < 2.0*mionizable/P[i].Mass) // prandom > this, won't be able to ionize anything interesting
            {
                mionized=0.0; total_m_ionizable += mionizable; h_i2=h_i*h_i;
                u_to_temp_fac = 0.59 * PROTONMASS / BOLTZMANN * GAMMA_MINUS1 * All.UnitEnergy_in_cgs / All.UnitMass_in_g;
                uion = HIIRegion_Temp / u_to_temp_fac;
                startnode = All.MaxPart; jnearest=-1; rnearest=1.0e10; dummy=0; NITER_HIIFB=0;
                
                do {
                    jnearest=-1; rnearest=1.0e10;
                    R_search = RHII; if(h_i>R_search) R_search=h_i;
                    numngb = ngb_treefind_variable_threads(pos, R_search, -1, &startnode, 0, &dummy, &dummy, &dummy, Ngblist);
                    if(numngb>0)
                    {
                        for(n = 0; n < numngb; n++)
                        {
                            j = Ngblist[n];
                            if(P[j].Type == 0 && P[j].Mass > 0)
                            {
                                dx = pos[0] - P[j].Pos[0]; dy = pos[1] - P[j].Pos[1]; dz = pos[2] - P[j].Pos[2];
#ifdef BOX_PERIODIC               /*  now find the closest image in the given box size  */
                                NEAREST_XYZ(dx,dy,dz,1);
#endif
                                r2 = dx * dx + dy * dy + dz * dz; r=sqrt(r2);
                                /* check whether the particle is already ionized */
                                already_ionized = 0; rho_j = Particle_density_for_energy_i(j);
                                if(SphP[j].InternalEnergy<SphP[j].InternalEnergyPred) {u=SphP[j].InternalEnergy;} else {u=SphP[j].InternalEnergyPred;}
                                if((SphP[j].DelayTimeHII > 0)||(u>uion)) already_ionized=1;
                                /* now, if inside RHII and mionized<mionizeable and not already ionized, can be ionized! */
                                do_ionize=0; prob=0;
                                if((r<=RHII)&&(already_ionized==0)&&(mionized<mionizable))
                                {
                                    m_effective = P[j].Mass*(SphP[j].Density/rho);
                                    // weight by density b/c of how the recombination rate in each particle scales
                                    m_available = mionizable-mionized;
                                    if(m_effective<=m_available) {
                                        do_ionize=1; prob = 1.001;
                                    } else {
                                        prob = m_available/m_effective; // determine randomly if ionized
                                        if(prandom < prob) do_ionize=1;
                                    } // if(m_effective<=m_available) {
                                    if(do_ionize==1)
                                    {
                                        SphP[j].InternalEnergy = uion; SphP[j].InternalEnergyPred = SphP[j].InternalEnergy; SphP[j].DelayTimeHII = dt; already_ionized = 1;
                                        SphP[j].Ne = 1.0 + 2.0*yhelium(j); /* fully ionized */
                                    } // if(do_ionize==1)
                                    mionized += prob*m_effective;
                                } // if((r<=RHII)&&(already_ionized==0)&&(mionized<mionizable))
                                
                                /* if nearest un-ionized particle, mark as such */
                                if((r<rnearest)&&(already_ionized==0)) {rnearest = r; jnearest = j;}
                            } // if(P[j].Type == 0 && P[j].Mass > 0)
                        } // for(n = 0; n < numngb; n++)
                    } // if(numngb>0)
                    
                    // if still have photons and jnearest is un-ionized
                    if((mionized<mionizable)&&(jnearest>=0))
                    {
                        j=jnearest; m_effective = P[j].Mass*(SphP[j].Density/rho); m_available = mionizable-mionized; prob=m_available/m_effective; do_ionize=0; if(prandom < prob) do_ionize=1;
                        if(do_ionize==1)
                        {
                            SphP[j].InternalEnergy = uion; SphP[j].InternalEnergyPred = SphP[j].InternalEnergy; SphP[j].DelayTimeHII = dt;
                            SphP[j].Ne = 1.0 + 2.0*yhelium(j); /* fully ionized */
                        } // if(do_ionize==1)
                        mionized += prob*m_effective;
                    } // if((mionized<mionizable)&&(jnearest>=0))
                    
                    /* now check if we have ionized sufficient material, and if not, iterate with larger regions until we do */
                    RHIImultiplier=1.10;
                    if(mionized < 0.95*mionizable)
                    {
                        /* ok, this guy did not find enough gas to ionize, it needs to expand its search */
                        if((RHII >= 30.0*RHII_initial)||(RHII>=RHIIMAX)||(NITER_HIIFB >= MAX_N_ITERATIONS_HIIFB))
                        {
                            /* we're done looping, this is just too big an HII region */
                            mionized = 1.001*mionizable;
                        } else {
                            /* in this case we're allowed to keep expanding RHII */
                            if(mionized <= 0)
                            {
                                RHIImultiplier = 2.0;
                            } else {
                                RHIImultiplier=pow(mionized/mionizable,-0.333);
                                if(RHIImultiplier>5.0) RHIImultiplier=5.0;
                                if(RHIImultiplier<1.26) RHIImultiplier=1.26;
                            } // if(mionized <= 0)
                            RHII *= RHIImultiplier; if(RHII>1.26*RHIIMAX) RHII=1.26*RHIIMAX;
                            startnode=All.MaxPart; // this will trigger the while loop to continue
                        } // if((RHII >= 5.0*RHII_initial)||(RHII>=RHIIMAX)||(NITER_HIIFB >= MAX_N_ITERATIONS_HIIFB))
                    } // if(mionized < 0.95*mionizable)
                    NITER_HIIFB++;
                } while(startnode >= 0);
                total_m_ionized += mionized; avg_RHII += RHII;
            } // if(prandom < 2.0*mionizable/P[i].Mass)
        } // if((P[i].Type == 4)||(P[i].Type == 2)||(P[i].Type == 3))
    } // for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    myfree(Ngblist);
    
    MPI_Reduce(&total_m_ionizing, &totMPI_m_ionizing, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_l_ionizing, &totMPI_l_ionizing, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_m_ionized, &totMPI_m_ionized, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avg_RHII, &totMPI_avg_RHII, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0)
    {
        if(totMPI_m_ionizing>0)
        {
            totMPI_avg_RHII /= totMPI_m_ionizing;
#ifndef IO_REDUCED_MODE
            printf("HII PhotoHeating: Time=%g: %g sources with L_tot/erg=%g ; M_ionized=%g ; <R_HII>=%g \n", All.Time,totMPI_m_ionizing,totMPI_l_ionizing,totMPI_m_ionized,totMPI_avg_RHII); fflush(stdout);
#endif
            fprintf(FdHIIHeating, "%lg %g %g %g %g \n",All.Time,totMPI_m_ionizing,totMPI_l_ionizing,totMPI_m_ionized,totMPI_avg_RHII);
        }
#ifdef IO_REDUCED_MODE
        if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin)
#endif
        {fflush(FdHIIHeating);}
    } // ThisTask == 0
} // void HII_heating_singledomain(void)
#endif // GALSF_FB_FIRE_RT_HIIHEATING
