// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// $Id$
// $URL$
// Maintainer: ndtrung

#ifndef __PAIR_EVALUATOR_DIPOLE_H__
#define __PAIR_EVALUATOR_DIPOLE_H__

#ifndef __HIPCC__
#include <string>
#endif

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif
#include "QuaternionMath.h"
#include "hoomd/VectorMath.h"
#include <iostream>
/*! \file EvaluatorPairMorseAniso.h
    \brief Defines the aniso morse
*/

// need to declare these class methods with __device__ qualifiers when building
// in nvcc.  HOSTDEVICE is __host__ __device__ when included in nvcc and blank
// when included into the host compiler
#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

class EvaluatorPairMorseAniso
    {
    public:
    struct param_type
        {
        Scalar D0;
        Scalar alpha;
        Scalar r0;
        Scalar w;
        Scalar kai;
        

#ifdef ENABLE_HIP
        //! Set CUDA memory hints
        void set_memory_hint() const
            {
            // default implementation does nothing
            }
#endif

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory
            allocation
        */
        HOSTDEVICE void load_shared(char*& ptr, unsigned int& available_bytes) const { }

        HOSTDEVICE param_type() : D0(0), alpha(0),r0(0), w(0), kai(0) {}//A(0), kappa(0) { }

#ifndef __HIPCC__

        param_type(pybind11::dict v)
            {
            D0 = v["D0"].cast<Scalar>();
            alpha = v["alpha"].cast<Scalar>();
            r0 = v["r0"].cast<Scalar>();
            w = v["w"].cast<Scalar>();
            kai = v["kai"].cast<Scalar>();
            }

        pybind11::object toPython()
            {
            pybind11::dict v;
            v["D0"] = D0;
            v["alpha"] = alpha;
            v["r0"]  =r0;
            v["w"] = w;
            v["kai"] = kai;
            return v;
            }

#endif
        }
#ifdef SINGLE_PRECISION
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

    // Nullary structure required by AnisoPotentialPair.
    struct shape_type
        {
        //vec3<Scalar> mu;

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory allocation
        */
        HOSTDEVICE void load_shared(char*& ptr, unsigned int& available_bytes) const { }

        HOSTDEVICE shape_type() : { }

#ifndef __HIPCC__
       

         shape_type(pybind11::object shape_params) { }

        pybind11::object toPython()
            {
            return pybind11::none();
            }
#endif // __HIPCC__

#ifdef ENABLE_HIP
        //! Attach managed memory to CUDA stream
        void attach_to_stream(hipStream_t stream) const { }
#endif
        };

    //! Constructs the pair potential evaluator
    /*! \param _dr Displacement vector between particle centers of mass
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _qi Quaternion of i^{th} particle
        \param _qj Quaternion of j^{th} particle
        \param _A Electrostatic energy scale
        \param _kappa Inverse screening length
        \param _params Per type pair parameters of this potential
    */
    HOSTDEVICE EvaluatorPairDipole(Scalar3& _dr,
                                   Scalar4& _quat_i,
                                   Scalar4& _quat_j,
                                   Scalar _rcutsq,
                                   const param_type& _params)
        : dr(_dr), rcutsq(_rcutsq), qi(_qi), qj(_qj), 
         D0(_params.D0), alpha(_params.alpha), r0(_params.r0), w(_params.w), kai(_params.kai)
        {
        }

    HOSTDEVICE void load_shared(char*& ptr, unsigned int& available_bytes) const
        {
        // No-op for this struct since it contains no arrays
        }

    //! uses diameter
    HOSTDEVICE static bool needsDiameter()
        {
        return false;
        }

    //! Whether the pair potential uses shape.
    HOSTDEVICE static bool needsShape()
        {
        return false;
        }

    //! Whether the pair potential needs particle tags.
    HOSTDEVICE static bool needsTags()
        {
        return false;
        }

    //! whether pair potential requires charges
    HOSTDEVICE static bool needsCharge()
        {
        return false;
        }

    //! Accept the optional diameter values
    /*! \param di Diameter of particle i
        \param dj Diameter of particle j
    */
    HOSTDEVICE void setDiameter(Scalar di, Scalar dj) { }

    //! Accept the optional shape values
    /*! \param shape_i Shape of particle i
        \param shape_j Shape of particle j
    */
    HOSTDEVICE void setShape(const shape_type* shapei, const shape_type* shapej) { }


    //! Accept the optional tags
    /*! \param tag_i Tag of particle i
        \param tag_j Tag of particle j
    */
    HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj) { }

    //! Accept the optional charge values
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    HOSTDEVICE void setCharge(Scalar qi, Scalar qj) { }


    //! Evaluate the force and energy
    /*! \param force Output parameter to write the computed force.
        \param pair_eng Output parameter to write the computed pair energy.
        \param energy_shift If true, the potential must be shifted so that
            V(r) is continuous at the cutoff.
        \param torque_i The torque exerted on the i^th particle.
        \param torque_j The torque exerted on the j^th particle.
        \return True if they are evaluated or false if they are not because
            we are beyond the cutoff.
    */
    HOSTDEVICE bool evaluate(Scalar3& force,
                             Scalar& pair_eng,
                             bool energy_shift,
                             Scalar3& torque_i,
                             Scalar3& torque_j)
        {
        vec3<Scalar> rvec(dr);

        Scalar rsq = dot(rvec, rvec);



        if (rsq > rcutsq)
            return false;

         
        Scalar r = fast::sqrt(rsq);
        vec3<Scalar> n_rvec(rvec / r);

-
        Scalar Exp_factor = fast::exp(-alpha * (r - r0));
        Scalar orien_factor =  (Scalar(1.0) + fast::exp(-w * (dot(n_rvec,q_i) - kai))) *
        (Scalar(1.0) + fast::exp(-w * ((dot(-n_rvec,q_j)-kai)));

        e = D0 * Exp_factor * (Exp_factor - Scalar(2.0));
        f = Scalar(2.0) * D0 * alpha * Exp_factor * (Exp_factor - Scalar(1.0)) / r;

        if (energy_shift)
            {
            Scalar rcut = fast::sqrt(rcutsq);
            Scalar Exp_factor_cut = fast::exp(-alpha * (rcut - r0));
            e -= D0 * Exp_factor_cut * (Exp_factor_cut - Scalar(2.0));}




        
        force = vec_to_scalar3(f / orien_factor);
        // without torque 
        torque_i = make_scalar3(0, 0, 0);
        torque_j = make_scalar3(0, 0, 0);
        //
        pair_eng = e / orien_factor;
        return true;
        }

#ifndef __HIPCC__
    //! Get the name of the potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return "morse_aniso";
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar3 dr;             //!< Stored vector pointing between particle centers of mass
    Scalar rcutsq;          //!< Stored rcutsq from the constructor
    Scalar4 q_i, q_j;        //!< Stored particle charges
    //Scalar4 quat_i, quat_j; //!< Stored quaternion of ith and jth particle from constructor
    // vec3<Scalar> mu_i;      /// Magnetic moment for ith particle
    // vec3<Scalar> mu_j;      /// Magnetic moment for jth particle
    Scalar D0;
    Scalar alpha;
    Scalar r0;
    Scalar w; 
    Scalar kai; //cos(\alpha_0)
    // const param_type &params;   //!< The pair potential parameters
    };

#endif // __PAIR_EVALUATOR_MORSE_ANISO_H__
