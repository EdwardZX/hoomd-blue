/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

// conditionally compile in only if boost is 1.35 or later
#include <boost/version.hpp>
#if (BOOST_VERSION >= 103500)

#include <iostream>
using namespace std;

#include <boost/python.hpp>
using namespace boost::python;

#include "ElectrostaticShortRange.h"
#include <stdexcept>
#include <boost/math/special_functions/erf.hpp>

// #define _USE_MATH_DEFINES
// #include <math.h>

//! This number is just 2/sqrt(Pi)
#define EWALD_F  1.128379167

//! This is a tolerance number for the spacing of the look up table
#define TOL 1e-6
// It is completely unlikely that m_delta is ever chosen < TOL


/*! \file ElectrostaticShortRange.cc
    \brief Contains the code for the ElectrostaticShortRange class
*/

using namespace std;

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param r_cut Cuttoff radius beyond which the force is 0
    \param alpha Split parameter of short vs long range electrostatics (see header file for more info)
    \param delta Spacing of the lookup table for erfc (see header file for more info)
    \param min_value minimum value expected to compute using the look up table
*/
ElectrostaticShortRange::ElectrostaticShortRange(boost::shared_ptr<SystemDefinition> sysdef,
                                                 boost::shared_ptr<NeighborList> nlist,
                                                Scalar r_cut,
                                                Scalar alpha,
                                                Scalar delta,
                                                Scalar min_value)
    : ForceCompute(sysdef), m_nlist(nlist), m_r_cut(r_cut),m_alpha(alpha),m_delta(delta),m_min_value(min_value)
    {
    assert(m_pdata);
    assert(m_nlist);
    
    if (r_cut < 0.0)
        {
        cerr << endl << "**Error! Negative r_cut in ElectrostaticShortRange makes no sense" << endl << endl;
        throw runtime_error("Error initializing ElectrostaticShortRange");
        }
    if (alpha < 0.0)
        {
        cerr << endl << "***Error! Negative alpha in ElectrostaticShortRange makes no sense" << endl << endl;
        throw runtime_error("Error initializing ElectrostaticShortRange");
        }
        
    if (m_delta<0)
        {
        cerr << endl << "***Error! delta must be positive in ElectrostaticShortRange" << endl << endl;
        throw runtime_error("Error initializing ElectrostaticShortRange");
        }
        
    if ((m_min_value<m_delta)||(m_min_value<TOL))
        {
        cerr << endl << 
        "***Error! min_value must be larger than m_delta or TOL in ElectrostaticShortRange, otherwise errors may occur"  
            << endl << endl;
        throw runtime_error("Error initializing ElectrostaticShortRange");
        }
        
    int N_points_l=static_cast<int>(ceil(m_r_cut/m_delta))+2; // We add a buffer of 2m_delta as we need to compute n+1,n+2
    int N_points=N_points_l*N_points_l;
    
    f_table = new Scalar[N_points];
    e_table = new Scalar[N_points];
    memset(f_table, 0, sizeof(Scalar)*N_points);
    memset(e_table, 0, sizeof(Scalar)*N_points);
    
    Scalar rsq=0;
    
    //compute the look up table
    
    for (int i=0; i<N_points; i++)
        {
        rsq=sqrt(i*m_delta*m_delta);
        if (rsq > TOL)
            {
            Scalar alrsq=m_alpha*rsq;
            Scalar erfc_al=boost::math::erfc(alrsq);
            f_table[i]=(Scalar(EWALD_F)*m_alpha*exp(-alrsq*alrsq)+erfc_al/rsq)/pow(rsq,2);
            e_table[i]=Scalar(0.5)*erfc_al/rsq;
            }
        }
    }

ElectrostaticShortRange::~ElectrostaticShortRange()
    {
    delete[] f_table;
    delete[] e_table;
    f_table=NULL;
    e_table=NULL;
    // deallocate memory
    }

void ElectrostaticShortRange::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);
    
    // start the profile
    if (m_prof) m_prof->push("Elec pair");
    
    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    
    // access the neighbor list
    const vector< vector< unsigned int > >& full_list = m_nlist->getList();
    
    // access the particle data
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    // sanity check
    assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
    
    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    // create a temporary copy of r_cut squared and delta squared
    Scalar r_cut_sq = m_r_cut * m_r_cut;
    Scalar delta_sq=m_delta*m_delta;
    
    // precalculate box lenghts for use in the periodic imaging
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    // tally up the number of forces calculated
    int64_t n_calc = 0;
    
    // need to start from a zero force, potential energy and virial
    // (MEM TRANSFER 5*N Scalars)
    memset(m_fx, 0, sizeof(Scalar)*arrays.nparticles);
    memset(m_fy, 0, sizeof(Scalar)*arrays.nparticles);
    memset(m_fz, 0, sizeof(Scalar)*arrays.nparticles);
    memset(m_pe, 0, sizeof(Scalar)*arrays.nparticles);
    memset(m_virial, 0, sizeof(Scalar)*arrays.nparticles);
    
    // for each particle
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        // access the particle's position and charge (MEM TRANSFER: 4 Scalars)
        Scalar xi = arrays.x[i];
        Scalar yi = arrays.y[i];
        Scalar zi = arrays.z[i];
        
        Scalar q_i=arrays.charge[i];
        
        // zero force, potential energy, and virial for the current particle
        Scalar fxi = 0.0;
        Scalar fyi = 0.0;
        Scalar fzi = 0.0;
        Scalar pei = 0.0;
        Scalar viriali = 0.0;
        
        // loop over all of the neighbors of this particle
        const vector< unsigned int >& list = full_list[i];
        const unsigned int size = (unsigned int)list.size();
        
        for (unsigned int j = 0; j < size; j++)
            {
            // increment our calculation counter
            n_calc++;
            
            // access the current neighbor (MEM TRANSFER 1 integer)
            unsigned int k = list[j];
            // sanity check
            assert(k < m_pdata->getN());
            
            // calculate dr (FLOPS: 3 / MEM TRANSFER 3 Scalars)
            Scalar dx = xi - arrays.x[k];
            Scalar dy = yi - arrays.y[k];
            Scalar dz = zi - arrays.z[k];
            
            // apply periodic boundary conditions (FLOPS: 9)
            if (dx >= box.xhi)
                dx -= Lx;
            else if (dx < box.xlo)
                dx += Lx;
                
            if (dy >= box.yhi)
                dy -= Ly;
            else if (dy < box.ylo)
                dy += Ly;
                
            if (dz >= box.zhi)
                dz -= Lz;
            else if (dz < box.zlo)
                dz += Lz;
                
            // start computing the force
            // calculate r squared (FLOPS: 5)
            Scalar rsq = dx*dx + dy*dy + dz*dz;
            
            if (rsq < r_cut_sq)
                {
                // access the charge of the neighbor (MEM TRANSFER: 1 Scalar)
                Scalar q_k = arrays.charge[k];
                
                // Interpolation table using the Newton-Gregory interpolation method
                // (FLOPS: 18 / MEM TRANSFER 6 SCALARS)
                Scalar sds=rsq/delta_sq;
                Scalar floor_sds=floor(sds);
                int k_ind=static_cast<int>(floor_sds);
                Scalar xi=sds-floor_sds;
                Scalar vk=f_table[k_ind];
                Scalar vk1=f_table[k_ind+1];
                Scalar vk2=f_table[k_ind+2];
                Scalar ek=e_table[k_ind];
                Scalar ek1=e_table[k_ind+1];
                Scalar ek2=e_table[k_ind+2];
                Scalar t1=vk+(vk1-vk)*xi;
                Scalar t2=vk1+(vk2-vk1)*(xi-Scalar(1.0));
                Scalar e1=ek+(ek1-ek)*xi;
                Scalar e2=ek1+(ek2-ek1)*(xi-Scalar(1.0));
                
                // compute the force magnitude divided by r (FLOPS: 6)
                Scalar forcemag_divr = q_i*q_k*(t1+(t2-t1)*xi*Scalar(0.5));
                
                // calculate the virial and the potential energy (FLOPS: 8)
                Scalar pair_eng = q_i*q_k*(e1+(e2-e1)*xi*Scalar(0.5));
                Scalar pair_virial = Scalar(1.0/6.0) * rsq * forcemag_divr;
                
                // add the force to the particle i (FLOPS: 8)
                fxi += dx*forcemag_divr;
                fyi += dy*forcemag_divr;
                fzi += dz*forcemag_divr;
                pei += pair_eng;
                viriali += pair_virial;
                
                // add the force to particle j if we are using the third law
                if (third_law)
                    {
                    // (FLOPS: 8 / MEM TRANSFER: 10 Scalars)
                    m_fx[k] -= dx*forcemag_divr;
                    m_fy[k] -= dy*forcemag_divr;
                    m_fz[k] -= dz*forcemag_divr;
                    m_pe[k] += pair_eng;
                    m_virial[k] += pair_virial;
                    }
                }
                
            }
        // add the force, potential energy, and virial to particle i
        // (FLOPS: 5 / MEM TRANSFER: 10 Scalars)
        m_fx[i] += fxi;
        m_fy[i] += fyi;
        m_fz[i] += fzi;
        m_pe[i] += pei;
        m_virial[i] += viriali;
        }
        
    m_pdata->release();
    
#ifdef ENABLE_CUDA
    // the force data is now only up to date on the cpu
    m_data_location = cpu;
#endif
    
    int64_t flops = n_calc * (3+9+5+18+6+8+8);
    if (third_law)  flops += n_calc*8;
    int64_t mem_transfer = m_pdata->getN() * (5 + 4 + 10)*sizeof(Scalar) + n_calc * ( (3+6)*sizeof(Scalar) + (1)*sizeof(unsigned int));
    if (third_law) mem_transfer += n_calc*10*sizeof(Scalar);
    if (m_prof) m_prof->pop(flops, mem_transfer);
    }
#undef EWALD_F
#undef TOL
#endif

#ifdef WIN32
#pragma warning( pop )
#endif

