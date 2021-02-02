#include <AMReX.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MLPoisson_3D_K.H>
#include <AMReX_MultiFab.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParmParse.H>
#include <AMReX_BLProfiler.H>

using namespace amrex;

void test ();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    test();

    amrex::Finalize();
}

////////////////////////////////////////
// pure cuda version of poisson gsrb

__global__ void mlpoisson_gsrb_cuda(Array4<Real> const& phi,
				    Array4<Real const> const& rhs,
				    Real dhx, Real dhy, Real dhz,
				    int nx, int ny, int nz,
				    int redblack, int nGhost) {

  extern volatile __shared__ Real shmem[];
  int i0 = blockIdx.x*blockDim.x;
  int j0 = blockIdx.y*blockDim.y;
  int k0 = blockIdx.z*blockDim.z;
  int sy = (blockDim.x+2*nGhost);
  int sz = (blockDim.y+2*nGhost)*sy;

  if (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0 &&
      blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
    printf("mlpoisson_gsrb_cuda : phi(0,0,0)=%1.16g, rhs(0,0,0)=%1.16g\n",phi(0,0,0),rhs(0,0,0));

  // for (int kk=threadIdx.z-nGhost; kk<blockDim.z+nGhost; kk+=blockDim.z) {
  //   int k = (kk+nGhost)*sz;

  //   for (int jj=threadIdx.y-nGhost; jj<blockDim.y+nGhost; jj+=blockDim.y) {
  //     int j = (jj+nGhost)*sy;

  //     for (int ii=threadIdx.x-nGhost; ii<blockDim.x+nGhost; ii+=blockDim.x) {
  // 	int index = k + j + ii+nGhost;

  // 	shmem[index] = phi(i0+ii, j0+jj, k0+kk);
  //     }  
  //   }  
  // } 
  // __syncthreads();

  // int i = i0 + threadIdx.x;
  // int j = j0 + threadIdx.y;
  // int k = k0 + threadIdx.z;
    
  // if ((i+j+k)%2==redblack) {
  //   constexpr Real omega = Real(1.15);
  //   const Real gamma = Real(-2.)*(dhx+dhy+dhz);
    
  //   int index = threadIdx.x + nGhost 
  //     + (threadIdx.y + nGhost)*sy
  //     + (threadIdx.z + nGhost)*sz;

  //   Real res = rhs(i,j,k) - gamma*shmem[index]
  //     - dhx*(shmem[index-1] + shmem[index+1])
  //     - dhy*(shmem[index-sy] + shmem[index+sy])
  //     - dhz*(shmem[index-sz] + shmem[index+sz]);
  //   phi(i,j,k) = shmem[index] + (omega/gamma) * res;
  // }
}

void test ()
{
    BL_PROFILE("main");
#if (AMREX_SPACEDIM < 3)
    return;
#endif

    amrex::Vector<int> ncells (AMREX_SPACEDIM, 128);
    amrex::Vector<int> indexing (AMREX_SPACEDIM, 0);
    int n_boxes_per_rank = 0;
    int max_grid_size = 128;
    amrex::Vector<int> cuda_grid_size (AMREX_SPACEDIM, 128);
    cuda_grid_size[0] = AMREX_GPU_MAX_THREADS;
    cuda_grid_size[1] = 1;
    cuda_grid_size[2] = 1;

    amrex::Print() << "cuda_grid_size = " << cuda_grid_size[0]
		   << " " << cuda_grid_size[1]
		   << " " << cuda_grid_size[2]
		   << std::endl;

    /* Read parameters from input file */
    ParmParse pp;
    pp.queryarr("ncells", ncells, 0, AMREX_SPACEDIM);
    pp.queryarr("indexing", indexing, 0, AMREX_SPACEDIM);
    pp.query("n_boxes_per_rank", n_boxes_per_rank);
    pp.query("max_grid_size", max_grid_size);
    pp.queryarr("cuda_grid_size", cuda_grid_size, 0, AMREX_SPACEDIM);

    amrex::Print() << "ncells = " << ncells[0]
		   << " " << ncells[1]
		   << " " << ncells[2]
		   << std::endl;

    amrex::Print() << "cuda_grid_size = " << cuda_grid_size[0]
		   << " " << cuda_grid_size[1]
		   << " " << cuda_grid_size[2]
		   << std::endl;

    IntVect lower(AMREX_D_DECL(0,0,0)); /// = ZeroVector();
    IntVect upper(AMREX_D_DECL(ncells[0]-1, ncells[1]-1, ncells[2]-1));
    IndexType typ({AMREX_D_DECL(indexing[0],indexing[1],indexing[2])});

    Box domain(Box(lower,upper,typ));
    BoxArray ba(domain);
    ba.maxSize(max_grid_size);

    // This defines the physical box, [-1,1] in each direction.
    RealBox real_box({AMREX_D_DECL(-1.0,-1.0,-0.5)},
		     {AMREX_D_DECL( 1.0, 1.0, 0.5)});

    // This says we are using Cartesian coordinates
    int coord = 0;

    // This sets the boundary conditions to be doubly or triply periodic
    Array<int,AMREX_SPACEDIM> is_periodic {AMREX_D_DECL(1,1,1)};

    // This defines a Geometry object
    Geometry geom(domain, real_box, coord, is_periodic);

    // Build the distirbution across processors
    DistributionMapping dm(ba);

    ////////////////////////////////////////
    // fill phi 
    int nGhost = 1;

    // API : mfab.defined(BoxArray, DistributionMapping, numComponents, numGhost)    
    MultiFab phi;
    phi.define(ba, dm, 1, nGhost);

    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
	FArrayBox& fab = phi[mfi];
	Array4<Real> const& a = fab.array();

        amrex::ParallelForRNG(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, RandomEngine const& engine) noexcept
        {
	  a(i,j,k) = amrex::Random(engine);
        });
        Gpu::streamSynchronize(); // because of arrs
    }

    ////////////////////////////////////////
    // fill the rhs

    MultiFab rhs;
    rhs.define(ba, dm, 1, 0);

    for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
	FArrayBox& fab = rhs[mfi];
	Array4<Real> const& a = fab.array();

        amrex::ParallelForRNG(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, RandomEngine const& engine) noexcept
        {
	  a(i,j,k) = amrex::Random(engine);
        });
        Gpu::streamSynchronize(); // because of arrs
    }

    ////////////////////////////////////////
    // Fill the periodic boundaries
    phi.FillBoundary(geom.periodicity());
    rhs.FillBoundary(geom.periodicity());

    // Get geometric parameters
    const Real* dxinv = geom.InvCellSize();
    AMREX_D_TERM(const Real dhx = dxinv[0]*dxinv[0];,
                 const Real dhy = dxinv[1]*dxinv[1];,
                 const Real dhz = dxinv[2]*dxinv[2];);

    amrex::Print() << "dhx = " << dhx << " dhy = " << dhy << " dhz = " << dhz << std::endl;

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);

    ////////////////////////////////////////
    // Call the simplified RB GS method
    for (MFIter mfi(phi,mfi_info); mfi.isValid(); ++mfi)
    {
      const Box& tbx = mfi.tilebox();
      const Box& vbx = mfi.validbox();

      Array4<Real      > const& phifab = phi.array(mfi);
      Array4<Real const> const& rhsfab = rhs.const_array(mfi);

      printf("phifab=%p, rhsfab=%p\n",phifab,rhsfab);

      int redblack=0;
      AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
      {
	mlpoisson_gsrb_new(thread_box, phifab, rhsfab, dhx, dhy, dhz, redblack);
      });

      redblack=1;
      AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
      {
	mlpoisson_gsrb_new(thread_box, phifab, rhsfab, dhx, dhy, dhz, redblack);
      });
    }

    cudaDeviceSynchronize();

    ////////////////////////////////////////
    // Call the pure cuda RB GS method
    for (MFIter mfi(phi,mfi_info); mfi.isValid(); ++mfi)
    {
      const Box& tbx = mfi.tilebox();
      const Box& vbx = mfi.validbox();

      Array4<Real      > const& phifab = phi.array(mfi);
      Array4<Real const> const& rhsfab = rhs.const_array(mfi);

      printf("phifab=%p, rhsfab=%p\n",phifab,rhsfab);
      
      dim3 block(cuda_grid_size[0],cuda_grid_size[1],cuda_grid_size[2]);
      int nbx = (ncells[0]+cuda_grid_size[0]-1)/cuda_grid_size[0];
      int nby = (ncells[1]+cuda_grid_size[1]-1)/cuda_grid_size[1];
      int nbz = (ncells[2]+cuda_grid_size[2]-1)/cuda_grid_size[2];
      amrex::Print() << "nbx = " << nbx << " nby = " << nby << " nbz = " << nbz << std::endl;

      dim3 grid(nbx,nby,nbz);
      size_t shmem_bytes = (cuda_grid_size[0]+2*nGhost)*(cuda_grid_size[1]+2*nGhost)*(cuda_grid_size[2]+2*nGhost)*sizeof(Real);
      amrex::Print() << shmem_bytes << " " << sizeof(Real) << std::endl;

      // redblack = 0
      mlpoisson_gsrb_cuda<<<grid,block,shmem_bytes>>>(phifab, rhsfab, dhx, dhy, dhz, ncells[0], ncells[1], ncells[2], 0, nGhost);

      // redblack = 1
      mlpoisson_gsrb_cuda<<<grid,block,shmem_bytes>>>(phifab, rhsfab, dhx, dhy, dhz, ncells[0], ncells[1], ncells[2], 1, nGhost);
    }

}
