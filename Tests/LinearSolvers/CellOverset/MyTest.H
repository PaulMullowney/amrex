#ifndef MY_TEST_H_
#define MY_TEST_H_

#include <AMReX_MLMG.H>

class MyTest
{
public:

    MyTest ();

    void solve ();
    void writePlotfile ();
    void initData ();

    int getNumTrials() { return num_trials; }

private:

    void readParameters ();
    void initGrids ();

    int n_cell = 128;
    int max_grid_size = 64;
    int nsolve_grid_size=64;

    std::string plot_file_name{"plot"};

    // For MLMG solver
    int verbose = 2;
    int bottom_verbose = 2;
    int max_coarsening_level = 30;
    int smooth_num_sweeps = 2;

    amrex::Geometry geom;
    amrex::BoxArray grids;
    amrex::DistributionMapping dmap;

    amrex::MultiFab phi;
    amrex::MultiFab rhs;
    amrex::MultiFab exact_phi;
    amrex::MultiFab acoef;
    amrex::MultiFab bcoef;

    amrex::Real ascalar = 1.e-3;
    amrex::Real bscalar = 1.0;

    int do_overset = 1;
    amrex::iMultiFab oversetmask;

    int use_hypre = 0;

    bool do_plots = true;
    int num_trials = 1;
    bool use_nsolve = false;

};

#endif
