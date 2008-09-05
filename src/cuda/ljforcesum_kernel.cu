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

// $Id$
// $URL$

#include "gpu_forces.h"
#include "gpu_pdata.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file ljforcesum_kernel.cu
	\brief Contains code for the LJ force sum kernel on the GPU
	\details Functions in this file are NOT to be called by anyone not knowing 
		exactly what they are doing. They are designed to be used solely by 
		LJForceComputeGPU.
*/

///////////////////////////////////////// LJ params

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;

//! Kernel for calculating lj forces
/*! This kerenel is called to calculate the lennard-jones forces on all N particles

	\param d_forces Device memory array to write calculated forces to
	\param pdata Particle data on the GPU to calculate forces on
	\param nlist Neigbhor list data on the GPU to use to calculate the forces
	\param coeffs Coefficients to the lennard jones force.
	\param coeff_width Width of the coefficient matrix
	\param r_cutsq Precalculated r_cut*r_cut, where r_cut is the radius beyond which forces are
		set to 0
	\param box Box dimensions used to implement periodic boundary conditions
	
	\a coeffs is a pointer to a matrix in memory. \c coeffs[i*coeff_width+j].x is \a lj1 for the type pair \a i, \a j.
	Similarly, .y is the \a lj2 parameter. The values in d_coeffs are read into shared memory, so 
	\c coeff_width*coeff_width*sizeof(float2) bytes of extern shared memory must be allocated for the kernel call.
	
	Developer information:
	Each block will calculate the forces on a block of particles.
	Each thread will calculate the total force on one particle.
	The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
extern "C" __global__ void calcLJForces_kernel(float4 *d_forces, gpu_pdata_arrays pdata, gpu_nlist_array nlist, float2 *d_coeffs, int coeff_width, float r_cutsq, gpu_boxsize box)
	{
	// read in the coefficients
	extern __shared__ float2 s_coeffs[];
	for (int cur_offset = 0; cur_offset < coeff_width*coeff_width; cur_offset += blockDim.x)
		{
		if (cur_offset + threadIdx.x < coeff_width*coeff_width)
			s_coeffs[cur_offset + threadIdx.x] = d_coeffs[cur_offset + threadIdx.x];
		}
	__syncthreads();
	
	// start by identifying which particle we are to handle
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= pdata.local_num)
		return;
	
	int pidx = idx + pdata.local_beg;
	
	// load in the length of the list
	int n_neigh = nlist.n_neigh[pidx];

	// read in the position of our particle. Texture reads of float4's are faster than global reads on compute 1.0 hardware
	float4 pos = tex1Dfetch(pdata_pos_tex, pidx);
	
	// initialize the force to 0
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	// loop over neighbors
	#ifdef ARCH_SM13
	// sm13 offers warp voting which makes this hardware bug workaround less of a performance penalty
	for (int neigh_idx = 0; __any(neigh_idx < n_neigh); neigh_idx++)
	#else
	for (int neigh_idx = 0; neigh_idx < nlist.height; neigh_idx++)
	#endif
		{
		if (neigh_idx < n_neigh)
		{
		int cur_neigh = nlist.list[nlist.pitch*neigh_idx + pidx];
		
		// get the neighbor's position
		float4 neigh_pos = tex1Dfetch(pdata_pos_tex, cur_neigh);
		
		// calculate dr (with periodic boundary conditions)
		float dx = pos.x - neigh_pos.x;
		float dy = pos.y - neigh_pos.y;
		float dz = pos.z - neigh_pos.z;
			
		dx -= box.Lx * rintf(dx * box.Lxinv);
		dy -= box.Ly * rintf(dy * box.Lyinv);
		dz -= box.Lz * rintf(dz * box.Lzinv);
			
		// calculate r^2
		float rsq = dx*dx + dy*dy + dz*dz;
		
		// calculate 1/r^2
		float r2inv;
		if (rsq >= r_cutsq)
			r2inv = 0.0f;
		else
			r2inv = 1.0f / rsq;

		// lookup the coefficients between this combination of particle types
		int typ_pair = __float_as_int(neigh_pos.w) * coeff_width + __float_as_int(pos.w);
		float lj1 = s_coeffs[typ_pair].x;
		float lj2 = s_coeffs[typ_pair].y;
	
		// calculate 1/r^6
		float r6inv = r2inv*r2inv*r2inv;
		// calculate the force magnitude / r
		float fforce = r2inv * r6inv * (12.0f * lj1  * r6inv - 6.0f * lj2);
			
		// add up the force vector components
		force.x += dx * fforce;
		force.y += dy * fforce;
		force.z += dz * fforce;
		force.w += r6inv * (lj1 * r6inv - lj2);
		}
		}
	
	// potential energy per particle must be halved
	force.w *= 0.5f;
	// now that the force calculation is complete, write out the result
	d_forces[pidx] = force;
	}


/*! \param d_forces Device memory to write forces to
	\param pdata Particle data on the GPU to perform the calculation on
	\param box Box dimensions (in GPU format) to use for periodic boundary conditions
	\param nlist Neighbor list stored on the gpu
	\param r_cutsq Precomputed r_cut*r_cut, where r_cut is the radius beyond which the 
		force is set to 0
	\param M Block size to execute
	
	\returns Any error code resulting from the kernel launch
	\note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()
*/
cudaError_t gpu_ljforce_sum(float4 *d_forces, gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_nlist_array *nlist, float2 *d_coeffs, int coeff_width, float r_cutsq, int M)
	{
	assert(pdata);
	assert(nlist);
	assert(d_coeffs);
	assert(coeff_width > 0);

    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)pdata->local_num/ (double)M), 1, 1);
    dim3 threads(M, 1, 1);

	// bind the texture
	cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata->pos, sizeof(float4) * pdata->N);
	if (error != cudaSuccess)
		return error;

    // run the kernel
    calcLJForces_kernel<<< grid, threads, sizeof(float2)*coeff_width*coeff_width >>>(d_forces, *pdata, *nlist, d_coeffs, coeff_width, r_cutsq, *box);

	#ifdef NDEBUG
	return cudaSuccess;
	#else
	cudaThreadSynchronize();
	return cudaGetLastError();
	#endif
	}

// vim:syntax=cpp
