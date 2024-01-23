find ./ -name "*.o" -type f -delete
rm GIZMO
module load "gsl"; \
module load "hdf5"; \
make;
