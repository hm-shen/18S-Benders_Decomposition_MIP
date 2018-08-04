'''
Description : Generate random two-stage stochastic linear programs with
            : complete recourse.
Date        : 06/23/2018
Author      : Haoming Shen
'''

import sys
import logging
import argparse
import numpy as np
import scipy.io as sio
from tester.probgen import Two_Stage_Generator

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def arg_parser():

    '''
    Description: parsing input arguments.
    '''

    # initialize parser
    parser = argparse.ArgumentParser(description='Test Benders Decomposition')

    # add arguments

    parser.add_argument("-o", "--outpath",
                        help="Output folder of these random problems.",
                        type=str, default="../data/")

    parser.add_argument("-r", "--coeffs_range",
                        help="Range of the random number generated.",
                        type=int, default=10)

    parser.add_argument("-ns", "--num_scenarios", help="number of scenarios.",
                        type=int, default=10)

    parser.add_argument("-md", "--master_dim",
                        help="dimension of the master problem.",
                        type=int, default=60)

    parser.add_argument("-sd", "--sub_dim",
                        help="dimension of the sub problem.",
                        type=int, default=50)

    parser.add_argument("-nc", "--num_constrs",
                        help="number of constraints in sub problem.",
                        type=int, default=25)

    parser.add_argument("-np", "--nprobs",
                        help="Number of tests to run.",
                        type=int, default=1)

    args = parser.parse_args()

    return args


def gen_prob(args, ind):

    np.set_printoptions(precision=3, linewidth=100)

    # generating a random problem
    logging.info('Generate a random problem ...')
    pg = Two_Stage_Generator(args.master_dim, args.sub_dim,
                             args.num_scenarios, args.num_constrs,
                             args.coeffs_range, degen=False)
    prob_data = pg.generate_problem(seed=None)
    print 'The optimal solution of this problem is: %.3f' % prob_data['ans']

    filename = args.outpath +\
               'prob_md%d_sd%d_nc%d_ns%d_r%d_num%d.mat' %\
               (args.master_dim, args.sub_dim, args.num_constrs,
                args.num_scenarios, args.coeffs_range, ind)

    sio.savemat(filename, prob_data)

    print 'Problem has been saved to %s' % filename


if __name__ == '__main__':

    '''
    Test Benders Decomposition with randomly generated problems.
    '''

    # set up for numpy precision
    np.set_printoptions(precision=4)

    logging.info('Parsing input arguments ...')
    args = arg_parser()
    logging.info('Input arguments have been parsed.')

    for ind in range(args.nprobs):
        gen_prob(args, ind)