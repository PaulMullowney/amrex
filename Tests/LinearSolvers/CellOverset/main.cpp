
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include "MyTest.H"

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        BL_PROFILE("main");
        MyTest mytest;
        for(int i=0;i<mytest.getNumTrials();++i) {
            mytest.initData();
            mytest.solve();
        }
        mytest.writePlotfile();
    }

    amrex::Finalize();
}
