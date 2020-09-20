#include <mpi.h>
#include "mpi_types.h"

MPI_Datatype DataTypeToMPIType<float>::value = MPI_FLOAT;
MPI_Datatype DataTypeToMPIType<double>::value = MPI_DOUBLE;
