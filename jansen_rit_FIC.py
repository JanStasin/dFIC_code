# -*- coding: utf-8 -*-
# ORIGINAL DOCUMENTATION:
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)


# DOCUMENTATION ABOUT THIS VERSION EDITS:
"""
Jansen-Rit and current Feedback Inhibitory Control implementation by J.Stasinski and D.Perdikis

"""


import math
import numpy
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, List, Range, Final
from tvb.simulator.models.base import ModelNumbaDfun


"""
Jansen-Rit and current Feedback Inhibitory Control implementation by...

"""


class wFICJansenRit(ModelNumbaDfun):

    # Define traited attributes for this model, these represent possible kwargs.
    A = NArray(
        label=":math:`A`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    B = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.05, hi=0.15, step=0.01),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.05]),
        domain=Range(lo=0.025, hi=0.075, step=0.005),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV].

        The usual value for this parameter is 6.0.""")

    nu_max = NArray(
        label=r":math:`\nu_{max}`",
        default=numpy.array([0.0025]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [s^-1].""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")

    J = NArray(
        label=":math:`J`",
        default=numpy.array([135.0]),
        domain=Range(lo=65.0, hi=1350.0, step=1.),
        doc="""Average number of synapses between populations.""")

    a_1 = NArray(
        label=r":math:`\alpha_1`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.""")

    a_2 = NArray(
        label=r":math:`\alpha_2`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop.""")

    a_3 = NArray(
        label=r":math:`\alpha_3`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.""")

    a_4 = NArray(
        label=r":math:`\alpha_4`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.""")

    p_min = NArray(
        label=":math:`p_{min}`",
        default=numpy.array([0.12]),
        domain=Range(lo=0.0, hi=0.12, step=0.01),
        doc="""Minimum input firing rate.""")

    p_max = NArray(
        label=":math:`p_{max}`",
        default=numpy.array([0.32]),
        domain=Range(lo=0.0, hi=0.32, step=0.01),
        doc="""Maximum input firing rate.""")

    mu = NArray(
        label=r":math:`\mu_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate.""")

    # FIC parameters:
    eta = NArray(
        label=":math:`eta `",
        default=numpy.array([0.0001]),
        domain=Range(lo=0.000001, hi=1, step=0.001),
        doc="""adaptation rate parameter for FIC""")

    y0_target = NArray(
        label=":math:`y0_target `",
        default=numpy.array([0.11]),
        domain=Range(lo=0.01, hi=1., step=0.001),
        doc="""Target y0: tunning target for FIC procedure""")

    tau_d = NArray(
        label=":math:`tau_d `",
        default=numpy.array([10.0]),
        domain=Range(lo=10, hi=100000., step=10),
        doc="""tau_d - corresponds to the time window over which y0 and y2 are averaged (10 = 10s)""")

  # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"y0": numpy.array([-1.0, 1.0]),
                 "y1": numpy.array([-500.0, 500.0]),
                 "y2": numpy.array([-50.0, 50.0]),
                 "y3": numpy.array([-6.0, 6.0]),
                 "y4": numpy.array([-20.0, 20.0]),
                 "y5": numpy.array([-500.0, 500.0]),
                 "y0_d": numpy.array([-1, 1]),
                 "y2_d": numpy.array([-50.0, 50.0]),
                 "wFIC": numpy.array([0.,  10.]),
                 "PSP": numpy.array([-1000.,  1000.])},
        doc="""The values for each state-variable should be set to encompass
    the expected dynamic range of that state-variable for the current
    parameters, it is used as a mechanism for bounding random initial
    conditions when the simulation isn't started from an explicit history,
    it is also provides the default range of phase-plane plots.""")

    state_variables = tuple(
        'y0 y1 y2 y3 y4 y5 y0_d y2_d wFIC PSP'.split())
    non_integrated_variables = ['PSP']
    _nvar = 10
    cvar = numpy.array([1, 2], dtype=numpy.int32)

    """variables of interest have been changed from the original version to
    to allow for y1-y2 state variable selection as well as to set the default to:
    ['y1-y2', 'y0', 'wFIC']
    """

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=(['y1-y2', 'y0', 'y1', 'y2', 'y3',
                 'y4', 'y5', 'y0_d', 'y2_d', 'wFIC', 'PSP']),
        default=(['PSP', 'y0', 'wFIC']))

    _PSP = None
    use_numba = True

    def update_derived_parameters(self):
        """
        When needed, this should be a method for calculating parameters that are
        calculated based on paramaters directly set by the caller. For example,
        see, ReducedSetFitzHughNagumo. When not needed, this pass simplifies
        code that updates an arbitrary models parameters -- ie, this can be
        safely called on any model, whether it's used or not.
        """
        self.n_nonintvar = self.nvar - self.nintvar
        self._PSP = None
        self._stimulus = 0.0

    def update_state_variables_before_integration(self, state_variables, coupling, local_coupling=0.0, stimulus=0.0):
        self._stimulus = stimulus
        if self.use_numba:
            state_variables = \
                _numba_update_non_state_variables_before_integration(
                    state_variables.reshape(state_variables.shape[:-1]).T)
            state_variables = state_variables.T[..., numpy.newaxis]
        else:
            #   PSP             =    y1         -         wFIC         *       y2
            state_variables[-1] = state_variables[1] - (state_variables[-2] * state_variables[2])
        # Keep a copy:
        self._PSP = state_variables[-1].copy()
        return state_variables

    def _integration_to_state_variables(self, integration_variables):
        return numpy.array(integration_variables.tolist() + [0.0*integration_variables[0]] * self.n_nonintvar)

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0, PSP=0.0):
        y0, y1, y2, y3, y4, y5, y0_d, y2_d, wFIC = state_variables  # Excluding the last position of PSP

        # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - wFIC * y_{2_j}]
        lrc = coupling[0, :]
        short_range_coupling = local_coupling*(y1 - wFIC * y2)

        exp = numpy.exp
        sigm_y1_y2 = 2.0 * self.nu_max / \
            (1.0 + exp(self.r * (self.v0 - PSP)))
        sigm_y0_1 = 2.0 * self.nu_max / \
            (1.0 + exp(self.r * (self.v0 - (self.a_1 * self.J * y0))))
        sigm_y0_3 = 2.0 * self.nu_max / \
            (1.0 + exp(self.r * (self.v0 - (self.a_3 * self.J * y0))))

        return numpy.array([
            y3,
            y4,
            y5,
            self.A * self.a * sigm_y1_y2 - 2.0 * self.a * y3 - self.a ** 2 * y0,
            self.A * self.a * (self.mu + self.a_2 * self.J *
                               sigm_y0_1 + lrc + short_range_coupling)
            - 2.0 * self.a * y4 - self.a ** 2 * y1,
            self.B * self.b * (self.a_4 * self.J * sigm_y0_3) -
            2.0 * self.b * y5 - self.b ** 2 * y2,
            (y0 - y0_d)/self.tau_d,  # (y[0] -y0_detect)/self.tau_d,
            (y2 - y2_d)/self.tau_d,
            self.eta * y2_d * (y0_d - self.y0_target)
        ])

    def dfun(self, y, c, local_coupling=0.0):
        if self._PSP is None:
            # This will be called usually for integration algorithms like Heun or Runge Kutta
            # that require more than one points.
            state_variables = self._integration_to_state_variables(y)
            state_variables = \
                self.update_state_variables_before_integration(state_variables, c, local_coupling, self._stimulus)
            PSP = state_variables[-1]  # PSP
        else:
            PSP = self._PSP
        if self.use_numba:
            # Variables (n_regions, n_svs):
            y_ = y.reshape(y.shape[:-1]).T
            c_ = c.reshape(c.shape[:-1]).T
            # As a parameter (n_regions, ):
            src = local_coupling*(y[1] - y[-2]*y[2])[:, 0]
            deriv = _numba_dfun_jr(y_, c_,
                                   PSP,  # PSP for as a variable, or PSP[:, 0] for as a parameter
                                   src,
                                   self.nu_max, self.r, self.v0, self.a, self.a_1, self.a_2, self.a_3, self.a_4,
                                   self.A, self.b, self.B, self.J, self.mu, self.eta, self.y0_target, self.tau_d)
            deriv = deriv.T[..., numpy.newaxis]
        else:
            deriv = self._numpy_dfun(y, c, local_coupling, PSP)
        #  Set _PSP to None so that it is recomputed on subsequent steps
        #  for multistep integration schemes such as Heun or Runge-Kutta:
        self._PSP = None
        return deriv


@guvectorize([(float64[:],) * 2], '(n)' + '->(n)', nopython=True)
def _numba_update_non_state_variables_before_integration(y, ynew):
    ynew[:-1] = y[:-1]  # All state variables remain the same, except for the last one, the PSP
    ynew[-1] = y[1] - y[-2] * y[2]  # PSP = y1 - wFIC * y2


# @guvectorize([(float64[:],) * 21], '(n),(m)' + ',()'*18 + '->(n)', nopython=True)     # For PSP parameter
@guvectorize([(float64[:],) * 21], '(n),(m),(k)' + ',()'*17 + '->(n)', nopython=True)  # For PSP variable
def _numba_dfun_jr(y, c,
                   PSP,
                   src,
                   nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu, eta, y0_target, tau_d,
                   dx):
    sigm_y1_y2 = 2.0 * nu_max[0] / (1.0 + math.exp(r[0] * (v0[0] - PSP[0])))
    sigm_y0_1 = 2.0 * \
        nu_max[0] / (1.0 + math.exp(r[0] * (v0[0] - (a_1[0] * J[0] * y[0]))))
    sigm_y0_3 = 2.0 * \
        nu_max[0] / (1.0 + math.exp(r[0] * (v0[0] - (a_3[0] * J[0] * y[0]))))
    dx[0] = y[3]
    dx[1] = y[4]
    dx[2] = y[5]
    dx[3] = A[0] * a[0] * sigm_y1_y2 - 2.0 * a[0] * y[3] - a[0] ** 2 * y[0]
    dx[4] = A[0] * a[0] * (mu[0] + a_2[0] * J[0] * sigm_y0_1 +
                           c[0] + src[0]) - 2.0 * a[0] * y[4] - a[0] ** 2 * y[1]
    dx[5] = B[0] * b[0] * (a_4[0] * J[0] * sigm_y0_3) - \
        2.0 * b[0] * y[5] - b[0] ** 2 * y[2]
    dx[6] = (y[0] - y[6]) / tau_d[0]
    dx[7] = (y[2] - y[7]) / tau_d[0]
    dx[8] = eta[0] * y[7] * (y[6] - y0_target[0])
