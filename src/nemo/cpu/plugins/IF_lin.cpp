#ifndef NEMO_CPU_PLUGINS_IF_LIN
#define NEMO_CPU_PLUGINS_IF_LIN

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file IF_lin.cpp Neuron update CPU kernel for current-based
 * exponential decay integrate-and-fire neurons. */

#include <cassert>

#include <nemo/fixedpoint.hpp>
#include <nemo/plugins/IF_lin.h>

#include "neuron_model.h"

#include <cstdio>
#include <cstdlib>
#define LOG(...) fprintf(stdout, __VA_ARGS__);

extern "C"
NEMO_PLUGIN_DLL_PUBLIC
void
cpu_update_neurons(
		unsigned start, unsigned end,
		unsigned cycle,
		float* paramBase, size_t paramStride,
		float* stateBase, size_t stateHistoryStride, size_t stateVarStride,
		unsigned fbits,
		unsigned fstim[],
		RNG rng[],
		float currentEPSP[],
		float currentIPSP[],
		float currentExternal[],
		uint64_t recentFiring[],
		unsigned fired[],
		void* /* rcm */)
{
	const float* p_c_m        = paramBase + PARAM_C_M        * paramStride;
	const float* p_tau_refrac = paramBase + PARAM_TAU_REFRAC * paramStride;
	const float* p_v_thresh   = paramBase + PARAM_V_THRESH   * paramStride;
	const float* p_I_offset   = paramBase + PARAM_I_OFFSET   * paramStride;

	const size_t historyLength = 1;

	/* Current state */
	size_t b0 = cycle % historyLength;
	const float* v0  = stateBase + b0 * stateHistoryStride + STATE_V * stateVarStride;
	// const float* Ie0 = stateBase + b0 * stateHistoryStride + STATE_IE * stateVarStride;
	// const float* Ii0 = stateBase + b0 * stateHistoryStride + STATE_II * stateVarStride;
	const float* lastfired0 = stateBase + b0 * stateHistoryStride + STATE_LASTFIRED * stateVarStride;

	/* Next state */
	size_t b1 = (cycle+1) % historyLength;
	float* v1  = stateBase + b1 * stateHistoryStride + STATE_V * stateVarStride;
	// float* Ie1 = stateBase + b1 * stateHistoryStride + STATE_IE * stateVarStride;
	// float* Ii1 = stateBase + b1 * stateHistoryStride + STATE_II * stateVarStride;
	float* lastfired1 = stateBase + b1 * stateHistoryStride + STATE_LASTFIRED * stateVarStride;

	/* Each neuron has two indices: a local index (within the group containing
	 * neurons of the same type) and a global index. */

	int nn = end-start;
	assert(nn >= 0);

#pragma omp parallel for default(shared)
	for(int nl=0; nl < nn; nl++) {

		unsigned ng = start + nl;

		//! \todo consider pre-multiplying tau_syn_E/tau_syn_I
		//! \todo use euler method for the decay as well?
		// float Ie = Ie0[nl] + currentEPSP[ng];
		// float Ii = Ii0[nl] + currentIPSP[ng];

		/* Update the incoming current */
		float I = currentEPSP[ng] + currentIPSP[ng] + currentExternal[ng] + p_I_offset[nl];
		/* LOG("ng: %u, nl: %u, currentEPSP: %f\n", ng, nl, currentEPSP[ng]);
		LOG("ng: %u, nl: %u, currentIPSP: %f\n", ng, nl, currentIPSP[ng]);
		LOG("ng: %u, nl: %u, currentExternal: %f\n", ng, nl, currentExternal[ng]);
		LOG("ng: %u, nl: %u, currentIPSP: %f\n", ng, nl, p_I_offset[nl]);
		LOG("ng: %u, nl: %u, I: %f\n", ng, nl, I); */
		// Ie1[nl] = Ie;
		// Ii1[nl] = Ii;
		
		// float Ie = Ie0[nl] + currentEPSP[ng];
		// float Ii = Ii0[nl] + currentIPSP[ng];

		/* Update the incoming current */
		// float I = Ie + Ii + currentExternal[ng] + p_I_offset[nl];
		// Ie1[nl] = Ie;
		// Ii1[nl] = Ii;

		/* no need to clear current?PSP. */

		//! \todo clear this outside kernel. Make sure to do this in all kernels.
		currentExternal[ng] = 0.0f;

		float v = v0[nl];
		bool refractory = lastfired0[nl] <= p_tau_refrac[nl];

		/* If we're in the refractory period, no internal dynamics */
		if(!refractory) {
			float c_m = p_c_m[nl];
			v += I / c_m;
			// if (v<0) v = 0;
		}

		/* Firing can be forced externally, even during refractory period */
		fired[ng] = (!refractory && v >= p_v_thresh[nl]) || fstim[ng] ;
		fstim[ng] = 0;
		recentFiring[ng] = (recentFiring[ng] << 1) | (uint64_t) fired[ng];

		if(fired[ng]) {
			// reset refractory counter
			//! \todo make this a built-in integer type instead
			lastfired1[nl] = 1;
			v1[nl] = 0;
			// LOG("c%lu: n%u fired\n", elapsedSimulation(), m_mapper.globalIdx(n));
		} else {
			lastfired1[nl] += 1;
			v1[nl] = v;
		}
	}
}


cpu_update_neurons_t* test = &cpu_update_neurons;


#include "default_init.c"

#endif
