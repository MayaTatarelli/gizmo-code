#include <stdio.h>
void output_compile_time_options(void)
{
printf(
"        PEBBLE_ACCRETION_TESTPROBLEM\n"
"        BOX_SPATIAL_DIMENSION=2\n"
"        BOX_SHEARING=2\n"
"        EOS_GAMMA=(1.001)\n"
"        EOS_ENFORCE_ADIABAT=(1.0)\n"
"        BOX_PERIODIC\n"
"        HYDRO_MESHLESS_FINITE_MASS\n"
"        SELFGRAVITY_OFF\n"
"        GRAIN_FLUID\n"
"        GRAIN_BACKREACTION\n"
"        GRAIN_RDI_TESTPROBLEM\n"
"        SPAWN_PARTICLES_VIA_SPLITTING\n"
"        BOX_BND_PARTICLES\n"
"        BOX_SHEARING_Q=(3./2.)\n"
"\n");
}
