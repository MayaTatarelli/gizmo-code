PEBBLE_ACCRETION_TESTPROBLEM # special flags/custom code just for this problem (search for it to find what may need to be modified)
#HYDRO_FIX_MESH_MOTION=4 # this fixes the gas cells on shearing orbits: allows more like a 'regular grid': better in low-density regions but less resolution at high-rho
BOX_SPATIAL_DIMENSION=3 # (optional: set=2,1 for 2D,1D box)
BOX_LONG_Z=2./3. #(for xy=12 and z=8) #4./3.(for xy=6 and z=8) #1./3. #2./5. #For LBox_xy=6 and LBox_z=2 -> 1./3. # (z direction only H wide) (if shearing box is 6H, then 1/3 = 2, so 1H above and below)
BOX_SHEARING=4 # shearing box boundaries: 1=r-z sheet (r,z,phi coordinates), 2=r-phi sheet (r,phi,z), 3=r-phi-z box, 4=as 3, with vertical gravity
EOS_GAMMA=(1.001) # eos (setting exactly=1 causes division by zero, so for 1->1.00001; for adiabatic set =7./4.)
EOS_ENFORCE_ADIABAT=(1.0) # enforce EOS even in shocks/viscosity (cool back to isotherm/barytrope; disable to allow entropy evolution)
# (options below here are unlikely to change, though of course possible to do so)
BOX_PERIODIC # box is periodic in dimensions not otherwise constrained
BOX_OUTFLOW_Z
#HYDRO_MESHLESS_FINITE_VOLUME # default to MFV hydro (allow mass fluxes)
HYDRO_MESHLESS_FINITE_MASS
SELFGRAVITY_OFF # no self-gravity
GRAIN_FLUID # enable dust particles (master switch)
GRAIN_BACKREACTION # include back-reaction of grains pushing on gas (necessary for these instabilities)
GRAIN_RDI_TESTPROBLEM # adds variables to read and ICs to appropriately set up the problem
SPAWN_PARTICLES_VIA_SPLITTING # master switch needed to allow particle-spawning at the boundary for this problem
BOX_BND_PARTICLES # allows existence of special boundary particles (needed for our 'spawners')
BOX_SHEARING_Q=(3./2.) # shearing box q=-dlnOmega/dlnr; will default to 3/2 (Keplerian) if not set
#
