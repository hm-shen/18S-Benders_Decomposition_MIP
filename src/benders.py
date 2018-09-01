'''
Description : Vanilla Benders decomposition implementation based on
            : python 2.7 and Gurobi python API.
Author      : Haoming Shen
Date        : 02/13/2018

This class handles classes of problems with the following form:
min  c^\top x + \sum_{i=1}^S p_i \theta_i
s.t. Ax \leq b
     f_i(x) \leq 0, i = 1,2,\ldots,m
     x \succeq 0, \theta \succeq 0,
     x \in Z

where Q_j(x) =
min  q^\top y
s.t. W y = h_j - T x
     y \succeq 0

dual of sub:
max  (h_j - T x)^\top \lambda
s.t. - W^\top \lambda \preceq q

note : W is a full row rank matrix;
     : h_j is a column vector;
     : T is a matrix;
     : c is a column vector;
     : \sum p_i = 1.

note : c \succeq 0;
     : p \succeq 0;
     : q \succeq 0.
'''

import gurobipy as gb
import sys
import time
import copy
import logging
import numpy as np
from sub import Benders_Subproblem


class Benders_Decomposition(object):

    '''Benders Decomposition

    Input data:
    ----------

    indata  : info for W, h_j, T, q, j = 1,2,\ldots,S

    Attributes:
    ----------

    states : Statistics for Benders Decomposition
           : cuts : cuts at each iteration.
           : upper bounds : upper bounds at each iteration.
           : lower bounds : lower bounds at each iteration.
           : m_optval : optimal value at each iteration.
           : m_opt_theta : optimal theta at each iteratino.
           : status :
           : s_optval : optimal value of each subproblem.
           : total_time :
           : cuts_status : cut status at each iteration.

    '''

    def __init__(self, indata, opt_gap=0.1, tol_gap=1e-4, max_iters=10000):

        logging.debug("Initializing Benders Decomposition ...")

        # Status constants
        self.STATUS = {'NOT_OPTIMAL': 1, 'INFEASIBLE': 2}

        # configurations
        self.maxiters = max_iters # max number of iteration
        self.tol_gap = tol_gap    # tolerence for each subproblem
        self.opt_gap = opt_gap    # optimality gap (10% by default)

        # init variables
        self.data = {}          # problem data
        self.subdata = None     # data send to sub problems
        self.variables = {}     # gurobi variables
        self.constraints = {}   # gurobi constrs
        self.results = {}       # store final results
        self.stats = {}         # store computational statistics
        self._init_benders_records()

        # init submodel
        self.submodel = None

        # load data and build model
        if indata is not None:
            self._load_data(indata)
            self._build_model()

        logging.debug("Benders Decomposition is initialized.")

    def load_data(self, indata):

        logging.debug('Loading data ...')

        self._load_data(indata)
        self._build_model()

        logging.debug('Data is loaded.')

    def get_stats(self):

        return (self.results['m_optval'],
                self.results['total_time'],
                self.results['total_cuts'],
                self.results['total_iters'],
                self.stats['m_optval'])

    def _init_benders_records(self):

        logging.debug("Initializing internal variables ...")

        # init statistics
        self.stats['cuts'] = {} # cuts at each iteration
        self.stats['upper_bounds'] = [gb.GRB.INFINITY]
        self.stats['lower_bounds'] = [-gb.GRB.INFINITY]
        self.stats['m_optval'] = []
        self.stats['m_opt_x'] = []
        self.stats['m_opt_theta'] = []
        self.stats['status'] = []
        self.stats['s_optval'] = {}
        self.stats['total_time'] = None
        self.stats['n_feas_cuts'] = 0
        self.stats['n_opt_cuts'] = 0

        # denotes whether a cut is active or not
        self.stats['cuts_status'] = {}

        # init results
        self.results['m_optval'] = None
        self.results['m_opt_x'] = None
        self.results['m_opt_theta'] = None
        self.results['status'] = None
        self.results['total_time'] = None

        logging.debug("Internal variables have been initialized.")

    def _load_data(self, input_data):

        logging.debug("Loading input data ...")

        # read input data -- master
        # self.data = copy.deepcopy(input_data)
        self.data['c'] = input_data['c']
        self.data['p'] = input_data['p'].flatten()
        self.data['A'] = input_data.get('A', None)
        self.data['b'] = input_data.get('b', None)
        # read input data -- sub
        self.data['W'] = input_data['W']
        self.data['T'] = input_data['T']
        self.data['q'] = input_data['q']
        self.data['H'] = input_data['H'] # S columns
        # read problem answer
        # self.data['ans'] = input_data.get('ans', None)

        # check input dimension
        logging.debug("Dimensionality check.")

        dim0, dim1 = self.data['W'].shape
        assert dim0 == self.data['T'].shape[0],"Error: different dim of T and W!"
        assert dim0 == self.data['H'].shape[0],"Error: different dim of H and W!"
        assert dim1 == self.data['q'].shape[0],"Error: different dim of q and W!"

        # get the dimension of the problem
        self.data['master_dim'] = self.data['c'].shape[0]
        self.data['num_scenarios'] = self.data['p'].shape[0]
        self.data['sub_dim'] = dim1
        self.data['sub_dual_dim'] = dim0

        logging.debug("Loading complete.")

    def _prepare_sub_data(self, scenario_id):

        logging.debug("Preparing data for sub problem ...")

        if self.subdata is None:
            # create subdata from scratch
            # subdata: x at the ith loop,
            self.subdata = {'T': self.data['T'],
                            'W': self.data['W'],
                            'q': self.data['q'],
                            'c': self.data['c'],
                            'p_s': self.data['p'][scenario_id],
                            'sub_dim': self.data['sub_dim'],
                            'sub_dual_dim': self.data['sub_dual_dim'],
                            'scenario_id': scenario_id,
                            'h_s': self.data['H'][:,scenario_id][:,np.newaxis],
                            'x': self.stats['m_opt_x'][-1]}
                            #'theta_s': self.stats['m_opt_theta'][-1][scenario_id]}
        else:
            # subdata already exists
            # modify scenario_id and h_s
            self.subdata['scenario_id'] = scenario_id
            self.subdata['h_s'] = self.data['H'][:,scenario_id][:,np.newaxis]
            self.subdata['x'] = self.stats['m_opt_x'][-1]
            self.subdata['p_s'] = self.data['p'][scenario_id]
            # self.subdata['theta'] = self.stats['m_opt_theta'][-1]


        logging.debug("Preparation is complete.")

    def _build_model(self):

        logging.debug("Building computational model ...")

        self.model = gb.Model()
        self.model.setParam(gb.GRB.Param.OutputFlag, 0)
        self.model.setParam(gb.GRB.Param.Presolve, 0)
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

        logging.debug("Computational model has been built.")

    def _build_variables(self):

        logging.debug("Building variabls for master problem ...")

        m = self.model
        # build x
        x_ls = [m.addVar(lb=0, vtype=gb.GRB.BINARY, name='x%d' % ind)
                for ind in range(self.data['master_dim'])]
        self.variables['x'] = dict(zip(range(len(x_ls)), x_ls))

        # build \theta
        theta_list = [m.addVar(lb=0, name='theta%d' % ind)
                      for ind in range(self.data['num_scenarios'])]

        self.variables['theta'] = dict(zip(range(len(theta_list)), theta_list))
        m.update()

        logging.debug('Variables of master problem are built.')

    def _build_objective(self):

        logging.debug("Building objective functions for master problem ...")

        self.objective = self.model.setObjective(
            gb.quicksum(float(self.data['c'][ind]) * self.variables['x'][ind]
                        for ind in range(self.data['master_dim'])) +
            gb.quicksum(float(self.data['p'][ind]) * self.variables['theta'][ind]
                        for ind in range(self.data['num_scenarios'])))

        self.model.update()

        logging.debug("Objective function of master problem has been built.")

    def _build_constraints(self):

        logging.debug("Building constraints for master problem ...")

        # add init constraints
        m = self.model

        self.constraints['master'] = []
        # add master constraints if A, b exists
        if (self.data['A'] is not None) and (self.data['b'] is not None):
            for cid in range(self.data['A'].shape[0]):
                self.constraints['master'].append(m.addConstr(
                    gb.quicksum(self.variables['x'][ind] * self.data['A'][cid, ind]
                                for ind in range(self.data['master_dim'])),
                    gb.GRB.LESS_EQUAL, float(self.data['b'][cid])))
        m.update()

        # init cuts
        self.constraints['cuts'] = {}
        for ind in range(self.data['num_scenarios']):
            ''' init cuts list for each scenario '''
            self.constraints['cuts']['s%d' % ind] = []

        m.update()

        logging.debug("Constraints for master problem have been built.")

    def optimize(self):

        logging.debug("Start optimizing ...")

        st = time.time()

        # flag for stopping
        notDone = True # done?
        iternum = 0    # keep track of the number of iteration

        # while not done.
        while notDone and (iternum < self.maxiters):

            logging.info("Loop {} starts ...".format(iternum))

            # initializing statistical vars
            self.stats['s_optval'][iternum] = []
            self.stats['cuts'][iternum] = {}
            self.stats['cuts_status'][iternum] = [0] * self.data['num_scenarios']

            # reset flag
            # allPass = True
            self.model.update()
            self.model.optimize()

            if self.model.status == gb.GRB.Status.OPTIMAL:
                logging.info('Optimal objective: %g' % self.model.objVal)
            elif self.model.status != gb.GRB.Status.INFEASIBLE:
                logging.info('Optimization was stopped with status %d' % self.model.status)
                exit(-1)
            # logging.info('Current primal optimal sol is: %.4f' % self.model.objVal)
            self._save_results(iternum)

            # debug
            logging.info("optval at {}th loop is {}"\
                         .format(iternum, self.stats['m_optval'][-1]))

            # if master is infeasible, so is the original problem
            if self.stats['status'][0] == gb.GRB.Status.INFEASIBLE:
                logging.error("Infeasible problem!")
                return np.nan

            if self.stats['status'][0] == gb.GRB.Status.UNBOUNDED:
                logging.error("Master is unbounded! Check whether c,p \succeq 0?")
                return np.nan

            st1 = time.time()
            # pre-select using cut selection strategy
            self._pre_select(iternum)

            # solve sub-problems
            allPass = self._check_subproblems(iternum)

            # all possible cuts are generated and
            # saved in self.stats['cuts'][iternum]
            # use selected mode to choose what to add.

            self._post_select(iternum)
            self._add_cuts(iternum)
            optgap = self._update_bounds(iternum)

            ed1 = time.time()
            logging.info('time to process: %.3f' % (ed1 - st1))
            logging.info('Total number of optimality cuts in model is:%d' %
                          self.stats['n_opt_cuts'])
            logging.info('Total number of feasibility cuts in model is:%d' %
                          self.stats['n_feas_cuts'])
            logging.info('Total number of constraints is model is: {}'
                         .format(self.model.NumConstrs))
            logging.info('current optgap is {}'.format(optgap))

            iternum += 1
            notDone = not allPass

            if (optgap >= 0) and (optgap < self.opt_gap):
                notDone = False
            else:
                notDone = True

        ed = time.time()
        self.stats['total_time'] = ed - st
        self.stats['total_iters'] = iternum
        print 'total time: %.2f' % (ed - st)
        self._prepare_results()
        logging.info("Optimization is complete.")

        return self.stats['m_optval'][-1]

    def _pre_select(self, iternum):

        # init dict for storing cuts info
        for sid in range(self.data['num_scenarios']):
            self.stats['cuts'][iternum][sid] = {}

    def _check_subproblems(self, iternum):

        all_pass = True

        for sid in range(self.data['num_scenarios']):

            self._solve_sub(iternum, sid)

            # generate benders cuts for this scenario
            flag_cut = self._get_cut(iternum, sid,
                                     self.submodel.results['status'])
            all_pass = all_pass and flag_cut

        return all_pass

    def _solve_sub(self, iternum, sid):

        logging.debug('Start solving sub problem ...')
        # construct subproblem for each scenario
        self._prepare_sub_data(sid)

        if self.submodel is None:
            self.submodel = Benders_Subproblem(self.subdata)
            self.submodel.optimize()
        else:
            self.submodel.update_model(self.subdata)
            self.submodel.reoptimize()

        if self.submodel.results['status'] == gb.GRB.Status.OPTIMAL:
            self.stats['s_optval'][iternum].append(
                self.submodel.results['s_optval'])
        elif self.submodel.results['status'] == gb.GRB.Status.INFEASIBLE:
            self.stats['s_optval'][iternum].append(-np.inf)
        elif self.submodel.results['status'] == gb.GRB.Status.UNBOUNDED:
            self.stats['s_optval'][iternum].append(np.inf)

        logging.debug('Sub problem is solved.')

    def _get_cut(self, iternum, sid, substatus):

        logging.debug("Getting benders cut for scenario %d" % sid)

        if substatus == gb.GRB.Status.OPTIMAL:
            logging.debug("sub problem {} has an optimal solution.".format(sid))
            pi = self.submodel.results['s_optsol']
            Q_i = self.submodel.results['s_optval']
            theta_i = self.stats['m_opt_theta'][-1][sid]
            h_s = self.data['H'][:, sid][:, np.newaxis]
            cut_id = len(self.constraints['cuts'])
            rhs_coeffs = pi.T.dot(self.data['T']).flatten()
            lhs = pi.T.dot(h_s)

            # save to stats
            self.stats['cuts'][iternum][sid]['status'] = ['optimal']
            self.stats['cuts'][iternum][sid]['pi'] = pi
            self.stats['cuts'][iternum][sid]['rhs_coeffs'] = rhs_coeffs
            self.stats['cuts'][iternum][sid]['lhs'] = lhs
            self.stats['cuts'][iternum][sid]['Q_i'] = Q_i
            self.stats['cuts'][iternum][sid]['theta_i'] = theta_i


            if (Q_i - self.tol_gap > theta_i):

                self.stats['n_opt_cuts'] += 1
                # cut is active at this sid
                logging.debug("sub problem {} generates an optimality cut".format(sid))
                self.stats['cuts'][iternum][sid]['status'].append(False)
                # self.stats['cuts'][iternum][sid]['selection'] = True
                self.stats['cuts_status'][iternum][sid] = self.STATUS['NOT_OPTIMAL']
                logging.debug('Status for this sub-problem is: %s' %
                             str(self.stats['cuts'][iternum][sid]['status'][1]))

                return False
            else:
                # this subproblem is optimal , no need to add cuts
                self.stats['cuts'][iternum][sid]['status'].append(True)
                # self.stats['cuts'][iternum][sid]['selection'] = False
                logging.debug('No need to add cut.')

                return True

        elif substatus == gb.GRB.Status.INFEASIBLE:

            # sub problem is infeasible, we need to add a feasibility cut
            logging.debug("sub problem {} is infeasible, we need to add a feasibility cut.".format(sid))
            self.stats['n_feas_cuts'] += 1
            unbdray = np.array(self.submodel.results['unbdray'])
            h_s = self.data['H'][:,sid][:,np.newaxis]
            rhs = unbdray.dot(h_s)
            lhs_coeffs = unbdray.dot(self.data['T'])
            theta_i = self.stats['m_opt_theta'][-1][sid]

            # save to stats
            self.stats['cuts'][iternum][sid]['status'] = ['infeasible', False]
            self.stats['cuts'][iternum][sid]['rhs'] = rhs
            self.stats['cuts'][iternum][sid]['lhs_coeffs'] = lhs_coeffs
            self.stats['cuts'][iternum][sid]['unbdray'] = unbdray
            self.stats['cuts'][iternum][sid]['Q_i'] = -np.inf
            self.stats['cuts'][iternum][sid]['theta_i'] = theta_i
            self.stats['cuts_status'][iternum][sid] = self.STATUS['INFEASIBLE']

            return False

        elif substatus == gb.GRB.Status.UNBOUNDED:

            logging.info("The original problem is unbounded")
            self.stats['cuts'][iternum][sid]['status'] = ['unbounded', True]

            return True

        else:
            logging.error('ERROR: Invalid substatus!')
            print('ERROR: Invalid substatus!')
            sys.exit(-1)

    def _add_cut(self, model, iternum, sid):

        if self.stats['cuts'][iternum][sid]['status'][0] == 'optimal':
            # add an optimality cut
            self._add_optimality_cut(
                model,
                sid,
                self.stats['cuts'][iternum][sid]['lhs'],
                self.stats['cuts'][iternum][sid]['rhs_coeffs'])

        elif self.stats['cuts'][iternum][sid]['status'][0] == 'infeasible':
            # add an feasibility cut
            self._add_feasibility_cut(
                model,
                sid,
                self.stats['cuts'][iternum][sid]['lhs_coeffs'],
                self.stats['cuts'][iternum][sid]['rhs'])

        else:
            logging.error('Selected cut is neither non-optimal nor infeasible!')
            sys.exit(-1)

    def _update_bounds(self, iternum):

        logging.debug('Updating bounds ...')

        # update lower bounds
        lowbnds = self._update_lbds(iternum)

        # update upper bounds
        if len(self.stats['s_optval'][iternum]) == self.data['num_scenarios']:

            upbnds = self._update_ubds(iternum)
            return (upbnds - lowbnds) / (lowbnds + 1)

        else:

            upbnds = np.inf
            self.stats['upper_bounds'].append(float(upbnds))
            return np.inf

    def _update_lbds(self, iternum):

        lowbnds = self.stats['m_optval'][-1]
        self.stats['lower_bounds'].append(lowbnds)

        return lowbnds

    def _update_ubds(self, iternum):

        upbnds = self.data['c'].transpose().\
                 dot(self.stats['m_opt_x'][-1]).flatten() +\
                 np.array(self.stats['s_optval'][iternum]).dot(self.data['p'])

        self.stats['upper_bounds'].append(float(upbnds))

        return upbnds

    def _add_cuts(self, iternum):

        '''
        Description: _add_cuts adds cuts accord. to the output of _post_select
        '''

        logging.debug('Adding cuts ...')

        # generate all cuts at each iteration
        for ind in range(self.data['num_scenarios']):
            if self.stats['cuts'][iternum][ind]['selection']:
                # Selection is true, this scenario needs to add cuts
                self._add_cut(self.model, iternum, ind)

        logging.debug('Cuts have been added.')

    def _post_select(self, iternum):

        '''
        Description: _post_select uses the iteration id as input argument and
        evaluate each subproblem and decides which subproblem should be solved.
        '''

        # vanilla benders decomposition rate cuts accord. to status
        for ind in range(self.data['num_scenarios']):
            self.stats['cuts'][iternum][ind]['selection']\
                = not self.stats['cuts'][iternum][ind]['status'][1]

    def _add_optimality_cut(self, model, sid, lhs, rhs_coeffs):

        logging.debug('Adding optimality cut ...')

        constr = model.addConstr(
            model.getVarByName('theta%d' % sid) +\
            gb.quicksum(model.getVarByName('x%d' % ind) * float(rhs_coeffs[ind])
                        for ind in range(self.data['master_dim'])),
            gb.GRB.GREATER_EQUAL, lhs)

        if self.model is model:
            self.constraints['cuts']['s%d' % sid].append(constr)

        model.update()

        logging.debug('Optimality cut has been added.')

    def _add_feasibility_cut(self, model, sid, lhs_coeffs, rhs):

        logging.debug('Adding feasibility cut ...')

        constr = model.addConstr(
            gb.quicksum(model.getVarByName('x%d' % ind) *\
                        float(lhs_coeffs[ind])
                        for ind in range(self.data['master_dim'])),
            gb.GRB.GREATER_EQUAL, rhs)

        if self.model is model:
            self.constraints['cuts']['s%d' % sid].append(constr)

        model.update()

        logging.debug('Feasibility cut has been added.')

    def _save_results(self, numiter):

        logging.debug("Saving optimization results ...")

        # save results accord. to the status of model
        if self.model.status == gb.GRB.Status.OPTIMAL:
            self.stats['status'].append(gb.GRB.Status.OPTIMAL)
            self.stats['m_optval'].append(self.model.objval)
            self.stats['m_opt_x'].append(
                np.array(
                    [self.model.getVarByName('x%d' % ind).x
                     for ind in range(self.data['master_dim'])]
                )[:,np.newaxis])
            self.stats['m_opt_theta'].append(
                np.array(
                    [self.model.getVarByName('theta%d' % ind).x
                     for ind in range(self.data['num_scenarios'])]
                )[:,np.newaxis])

            logging.debug("Optimization statistics have been saved.")

        elif self.model.status == gb.GRB.Status.UNBOUNDED:

            self.stats['status'].append(gb.GRB.Status.UNBOUNDED)
            logging.error("Master problem is unbounded.")
            sys.exit(-1)

        elif self.model.status == gb.GRB.Status.INFEASIBLE:

            self.stats['status'].append(gb.GRB.Status.INFEASIBLE)
            logging.error("Master problem is infeasible.")
            print("Master problem is infeasible.")
            sys.exit(-1)

        else:
            print "Unkown status!"
            logging.error("Unkown status!")
            sys.exit(-1)

    def _prepare_results(self):

        logging.debug('Benders iteration is complete! Preparing results ...')

        # save optimal value at each iteration
        self.results['m_optvals'] = self.stats['m_optval']

        # save the last optval
        self.results['m_optval'] = self.stats['m_optval'][-1]
        self.results['m_opt_x'] = self.stats['m_opt_x'][-1]
        self.results['m_opt_theta'] = self.stats['m_opt_theta'][-1]
        self.results['lower_bd'] = self.stats['lower_bounds'][-1]
        self.results['upper_bd'] = self.stats['upper_bounds'][-1]
        self.results['total_time'] = self.stats['total_time']
        self.results['total_cuts'] = self.model.numConstrs
        self.results['total_iters'] = len(self.stats['m_optval'])
        self.results['n_opt_cuts'] = self.stats['n_opt_cuts']
        self.results['n_feas_cuts'] = self.stats['n_feas_cuts']

        if self.stats['status'][-1] == 2:
            self.results['status'] = 'Optimal'
        elif self.stats['status'][-1] == 3:
            self.results['status'] = 'Infeasible'
        elif self.stats['status'][-1] == 5:
            self.results['status'] = 'Unbounded'

        logging.debug('Results are prepared.')

    def get_results(self):

        return copy.deepcopy(self.results)

    def print_results(self):

        # print results to screen
        logging.debug("Printing results to screen ...")
        print '\nOptimization Results:'
        print '{0:20}{1:<10}'.format('Solving Status:', self.results['status'])
        print '{0:20}{1:<.4f}'.format('Optimal Sol.:', self.results['m_optval'])
        print '{0:20}{1:<.4f}'.format('Num Optimal Cuts:', self.results['n_opt_cuts'])
        print '{0:20}{1:<.4f}'.format('Num Feasib Cuts:', self.results['n_feas_cuts'])
        print '{0:20}{1:<.4f}'.format('Lower Bound:', self.results['lower_bd'])
        print '{0:20}{1:<.4f}'.format('Upper bound:', self.results['upper_bd'])
        print '{0:20}{1:<10}'.format('Total Cuts:', self.results['total_cuts'])
        print '{0:20}{1:<.4f}s'.format('Total Time:', self.results['total_time'])

        logging.debug("Results have been printed.")

    def write_to_disk(self, out_path):

        logging.debug("Writing results to disk ...")

        # create file handle
        report = open('%sbenders_report.txt' % out_path, 'w')

        report.write('\nOptimization Results:\n')
        report.write('{0:20}{1:<10}\n'.\
                     format('Solving status:', self.results['status']))
        report.write('{0:20}{1:<.4f}\n'.\
                     format('Optimal sol.:', self.results['m_optval']))
        report.write('{0:20}{1:<.4f}\n'.\
                     format('Lower bound:', self.results['lower_bd']))
        report.write('{0:20}{1:<.4f}\n'.\
                     format('Upper bound:', self.results['upper_bd']))
        report.write('{0:20}{1:<10}\n'.\
                     format('Total cuts:', self.results['total_cuts']))
        report.write('{0:20}{1:<.4f}s\n'.\
                     format('Total time:', self.results['total_time']))
        report.write('End.\n')
        report.write('\n\nBounds:\n')
        logging.debug("Results have been written.")
