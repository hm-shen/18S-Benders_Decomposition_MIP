'''
Description : This file uses gurobi to solve the sub problem of benders decomposition
Author      : Haoming Shen
Date        : 02/13/2018
Reference   : https://github.com/TueVJ/PyGuEx

This class solves the following optimization problem.

\max_\pi & \pi^\top (h_s - T x)
\st      & \pi^\top W \leq q
'''

import gurobipy as gb
import sys
import logging
import numpy as np


class Benders_Subproblem(object):
    '''Benders Decomposition Solver for Subproblems

    inputs:
    ----------
    subdata : h_s, T, x at the ith loop, q, sub_dim, sub_dual_dim, senario_id
            : note this senario_id stats from 1 (not 0)

    '''

    def __init__(self, subdata):

        logging.debug("Initializing sub problem ...")

        self.data = subdata
        self.stats = {}
        self.variables = {}
        self.constraints = {}
        self.results = {}

        logging.debug("Sub problem has been initialized.")

    def optimize(self):

        logging.debug("Start optimizing sub problem ...")

        self._build_model()
        self.model.optimize()
        self._save_records()

        logging.debug("Optimization is complete ...")

    def _save_records(self):

        if self.model.status == gb.GRB.Status.OPTIMAL:

            logging.debug("Dual of sub problem has optimal solution.")

            # dual of subproblem works OK
            self.results['status'] = gb.GRB.Status.OPTIMAL
            self.results['s_optval'] = self.model.objval
            self.results['s_optsol'] = np.array(
                [soli.x for soli in self.model.getVars()])[:,np.newaxis]
            self.results['unbdray'] = None

        elif self.model.status == gb.GRB.Status.UNBOUNDED:

            logging.debug("Dual of sub problem is unbounded.")

            # dual of sub is unbounded ==> primal of sub is infeasible
            self.results['status'] = gb.GRB.Status.INFEASIBLE
            self.results['s_optval'] = - np.inf
            self.results['s_optsol'] = np.nan
            self.results['unbdray'] = self.model.getAttr('UnbdRay')

            logging.debug("unbounded ray direction is {}".\
                          format(self.results['unbdray']))

        elif self.model.status == gb.GRB.Status.INFEASIBLE:

            logging.debug("Dual of sub problem is INFEASIBLE.")

            # dual of sub is infeasible ==> primal of sub is unbounded
            self.results['status'] = gb.GRB.Status.UNBOUNDED
            self.results['s_optval'] = np.inf
            self.results['s_optsol'] = np.nan

    def _build_model(self):

        self.model = gb.Model()
        self.model.setParam(gb.GRB.Param.OutputFlag, 0)
        self.model.setParam(gb.GRB.Param.InfUnbdInfo, 1)
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):

        logging.debug('Building variables for subproblem ...')

        m = self.model

        # build pi
        sub_pi_ls = [m.addVar(lb=-gb.GRB.INFINITY, name='pi%d' % ind) \
                     for ind in range(self.data['sub_dual_dim'])]

        self.variables['pi'] = dict(zip(range(len(sub_pi_ls)), sub_pi_ls))
        m.update()

        logging.debug('Variables are built.')

    def _build_objective(self):

        ''' set objective for supb prob '''
        logging.debug('Constructing objective function ...')

        m = self.model

        logging.debug('h_s of sub problem is: %s' % str(self.data['h_s']))
        logging.debug('W of subproblem is: %s' % str(self.data['W']))
        logging.debug('T of subproblem is: %s' % str(self.data['T']))

        objcoeff = self.data['h_s'] - self.data['T'].dot(self.data['x'])

        logging.debug('objective coeffs are:\n%s' % str(objcoeff))

        self.model.setObjective(
            gb.quicksum(self.variables['pi'][ind] * float(objcoeff[ind])
                        for ind in range(self.data['sub_dual_dim'])),
            gb.GRB.MAXIMIZE)

        m.update()

        logging.debug('Objective function has been constructed.')

    def _build_constraints(self):

        logging.debug('Adding constraints for sub problem ...')

        m = self.model

        # add affine inequality constraints to sub problem
        for ind in range(self.data['sub_dim']):
            W_icol = self.data['W'][:,ind]

            logging.debug('W_icol are:\n%s' % str(W_icol))

            self.constraints[ind] = m.addConstr(
                gb.quicksum(self.variables['pi'][jnd] * float(W_icol[jnd])
                            for jnd in range(self.data['sub_dual_dim'])),
                gb.GRB.LESS_EQUAL, float(self.data['q'][ind]))

        m.update()

        logging.debug('Constraints for sub problem have been added.')
