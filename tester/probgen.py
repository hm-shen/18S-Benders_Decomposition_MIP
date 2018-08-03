'''
Description : This file contains a two-stage problem generator
Author      : Haoming Shen
'''

import gurobipy as gb
import logging
import numpy as np


class Two_Stage_Generator(object):
    ''' This class generates two-stage optimization problems '''

    def __init__(self, master_dim, sub_dim, num_scenarios, num_constrs,
                 coeffs_range, with_ans=True, degen=False):
        '''
        Description : Initialize problem generator with dimension of the master
                    : optimization variable
        '''
        # dimension of the optimization variable in the master problem
        self.master_dim = master_dim
        # dimension of the optimization variable in the sub problem
        self.sub_dim = sub_dim
        # number of scenarios
        self.n_scenarios = num_scenarios
        # number of constraints
        self.n_constrs = num_constrs
        # range of the coeffcients
        self.c_range = coeffs_range
        # return answer or not and generate degenerate problems or not
        self.flags = {'ans': with_ans, 'degen': degen}

        # ----------------------------------------------------------------------
        # print current flags (for debugging & double checking)
        # ----------------------------------------------------------------------

        print 'WITH_ANS: %s \nDEGENERATE: %s \n' %\
            (str(self.flags['ans']), str(self.flags['degen']))

    def generate_problem(self, seed=None):
        '''
        Description : Randomly generate a two-stage optimization problem with
                    : complete recourse
        '''
        prob_data = self._gen(seed=seed)

        if self.flags['ans']:
            ans, _ = self._solve(prob_data)
            prob_data['ans'] = ans
            if np.isnan(ans):
                print prob_data

        return prob_data

    def _dec2bin(self, dec, length):

        bin_str = bin(dec)
        bin_out = np.zeros((length))
        bin_raw = np.fromiter(bin_str[2:], int)
        bin_out[-bin_raw.shape[0]:] = bin_raw
        return bin_out

    def _dec2binlist(self, lsOfDec, length):

        # print 'lsOfDec is:', lsOfDec.shape[0]

        lsOfBin = np.empty([lsOfDec.shape[0], length])
        cnt = 0
        for dec in lsOfDec:
            lsOfBin[cnt, :] = self._dec2bin(dec, length)
            # print self._dec2bin(dec, length)
            cnt += 1

        return lsOfBin

    def _gen(self, seed=None):

        '''
        Description: Generate complete two-stage benders problems
        '''

        logging.debug('Start generating random problem ...')

        rand_int = np.random.randint

        if seed is not None:
            np.random.seed(seed)

        # generate a probability distribution
        prob = np.random.random((self.n_scenarios,)) + 0.001
        # prob = np.random.standard_normal((self.n_scenarios,)) + 0.001
        prob = prob / np.sum(prob)

        # generate coefficients of the master problem
        c = self.c_range * np.random.random((self.master_dim,1))
        # q = self.c_range * (np.random.random((self.sub_dim,1)) + 0.001)
        q = np.random.random((self.sub_dim,1)) + 0.1
        # q = q * (np.array(range(self.sub_dim))[:, None] + 1)

        # Generate a complete recourse matrix
        # Generate a set of basis for space with dimension
        # (self.n_constrs + 1)
        tmp_basis = np.random.random((self.n_constrs+1, self.n_constrs+1))
        # calculate its center
        tmp_center = np.sum(tmp_basis, axis=1)
        unit_center = (tmp_center / np.linalg.norm(tmp_center))[:, None]
        # project this set of tmp_basis to the orthocomplement
        # subspace of tmp_center
        projected_basis = (np.eye(self.n_constrs+1) -
                           unit_center.dot(unit_center.T)).dot(tmp_basis)

        print 'rank of matrix is:', np.linalg.matrix_rank(projected_basis)

        # projected_basis, _ = np.linalg.qr(projected_basis)
        subspace_basis, S, _ = np.linalg.svd(projected_basis, full_matrices=False)
        print 'Singular values are:\n', S
        # select first n-1 columns of projected_basis
        subspace_basis = subspace_basis[:, :-1]
        W = subspace_basis.T.dot(projected_basis)
        # print 'W is:', W

        if self.flags['degen']:
            # generate degenerate recourse matrix W
            W = np.append(W, W.dot(
                2 * np.random.random((self.n_constrs+1,
                                      self.sub_dim-self.n_constrs-1)) - 0.5), axis=1)
        else:
            # just randomly generate the rest of W
            W = np.append(W, 2 * np.random.random(
                (self.n_constrs, self.sub_dim-self.n_constrs-1)) - 0.5, axis=1)

        print 'Shape of W is:', W.shape

        W = W / np.linalg.norm(W, axis=0)

        H = rand_int(1, self.c_range) * 2 * (np.random.random(
            (self.n_constrs, self.n_scenarios)) - 0.5)

        H = H * (np.array(range(self.n_scenarios))[::-1] + 1) * 0.01

        T = rand_int(1, self.c_range) * 2 * (np.random.random(
            (self.n_constrs, self.master_dim)) - 0.5)

        # collect randomly generated data
        prob_data = {'c': np.around(c, decimals=4),
                     'p': np.around(prob, decimals=5),
                     'q': np.around(q, decimals=4),
                     'W': np.around(W, decimals=4),
                     'H': np.around(H, decimals=4),
                     'T': np.around(T, decimals=4)}

        logging.debug('Random problem is generated.')

        return prob_data

    def _gen_noncomplete(self, seed=None):

        '''
        Description: Generate non-complete two-stage problem.
        '''

        logging.debug('Start generating random problem ...')

        # random integer generator
        rand_int = np.random.randint

        # set random seeds
        if seed is not None:
            np.random.seed(seed)

        # generate a probability distribution
        prob = np.random.random((self.n_scenarios,)) + 0.001
        prob = prob / np.sum(prob)

        # generate coefficients of the master problem
        c = np.random.random((self.master_dim,1))
        q = self.c_range * (np.random.random((self.sub_dim,1)) + 0.001)
        W = np.random.random((self.n_constrs, self.sub_dim))
        # W = W / np.linalg.norm(W, axis=0)

        H = rand_int(1, self.c_range) * (np.random.random((self.n_constrs, self.n_scenarios)) - 0.5)
        T = rand_int(1, self.c_range) * (np.random.random((self.n_constrs, self.master_dim)) - 0.5)

        # collect randomly generated data
        prob_data = {'c': np.around(c, decimals=1),
                     'p': np.around(prob, decimals=3),
                     'q': np.around(q, decimals=1),
                     'W': np.around(W, decimals=1),
                     'H': np.around(H, decimals=1),
                     'T': np.around(T, decimals=1)}

        logging.debug('Random problem is generated.')

        return prob_data




    # def _gen_complete1(self, seed=None):
    #     '''
    #     Description: Generate complete two-stage problem
    #     '''

    #     logging.debug('Start generating random problem ...')

    #     # random integer generator
    #     rand_int = np.random.randint

    #     # set random seeds
    #     if seed is not None:
    #         np.random.seed(seed)

    #     # generate a probability distribution
    #     prob = np.random.random((self.n_scenarios,)) + 0.001
    #     prob = prob / np.sum(prob)

    #     # generate coefficients of the master problem
    #     c = np.random.random((self.master_dim,1))
    #     q = self.c_range * (np.random.random((self.sub_dim,1)) + 0.001)
    #     W = np.random.random((self.n_constrs, self.sub_dim))
    #     tmp = np.array(range(2 ** self.n_constrs))
    #     weights = self._dec2binlist(tmp, self.n_constrs)
    #     weights[weights == 0] = -1
    #     W[:, :(2 ** self.n_constrs)] = W[:, :(2 ** self.n_constrs)] *\
    #                                    (weights.T)
    #     # W = W / np.linalg.norm(W, axis=0)

    #     H = rand_int(1, self.c_range) * (np.random.random((self.n_constrs, self.n_scenarios)) - 0.5)
    #     T = rand_int(1, self.c_range) * (np.random.random((self.n_constrs, self.master_dim)) - 0.5)

    #     # collect randomly generated data
    #     prob_data = {'c': np.around(c, decimals=1),
    #                  'p': np.around(prob, decimals=3),
    #                  'q': np.around(q, decimals=1),
    #                  'W': np.around(W, decimals=1),
    #                  'H': np.around(H, decimals=1),
    #                  'T': np.around(T, decimals=1)}

    #     logging.debug('Random problem is generated.')

    #     return prob_data

    def _solve(self, data):
        '''
        Use gurobi to solve the random problem generated.
        '''

        # get problem dimension
        num_scenarios = data['p'].shape[0]
        master_dim = data['c'].shape[0]
        sub_dim = data['q'].shape[0]

        # initialize gurobi solver
        m = gb.Model("general_benders")
        m.setParam(gb.GRB.Param.OutputFlag, 0)

        # build variable dict
        varx = {ind: m.addVar(lb=0, vtype=gb.GRB.INTEGER, name='x%d' % ind)
                for ind in range(master_dim)}

        vary = {}

        for sid in range(num_scenarios):
            for ind in range(sub_dim):
                vary[(sid,ind)] = m.addVar(lb=0, name='y%d%d' % (sid, ind))
        m.update()

        # build obj func
        m.setObjective(
            gb.quicksum(float(data['c'][ind])*varx[ind]\
                        for ind in range(master_dim)) +\
            gb.quicksum(float(data['p'][sid]) *\
                        gb.quicksum(float(data['q'][ind])*vary[(sid,ind)]
                                    for ind in range(sub_dim))
                        for sid in range(num_scenarios)))
        m.update()

        # build constraints
        for sid in range(num_scenarios):
            W = data['W']
            T = data['T']
            h = data['H'][:,sid]
            for ind in range(data['W'].shape[0]):
                m.addConstr(
                    gb.quicksum(float(W[ind,jnd])*vary[(sid,jnd)] for jnd in range(sub_dim)) +\
                    gb.quicksum(float(T[ind,jnd])*varx[jnd] for jnd in range(master_dim)),
                    gb.GRB.EQUAL, h[ind])
        m.update()
        m.optimize()

        if m.status == gb.GRB.Status.OPTIMAL:
            print('Optimal objective: %g' % m.objVal)
            return m.objVal, m
        elif m.status == gb.GRB.Status.INFEASIBLE:
            print('Optimization was stopped with status infeasible')
            return np.nan, np.nan
        elif m.status == gb.GRB.Status.UNBOUNDED:
            print('Optimization was stopped with status unbounded')
            return np.nan, np.nan
        else:
            print('Unkown status!')
            return np.nan, np.nan