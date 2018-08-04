'''
Description: Driver for Benders Decomposition for MIP
Date: 08/03/2018
Author: Haoming Shen
'''

import sys
import time
import logging
import argparse
import numpy as np
import scipy.io as sio

from src.benders import Benders_Decomposition

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def arg_parser():

    parser = argparse.ArgumentParser(
        description='Driver for MIP Benders Decomposition')

    # add arguments
    parser.add_argument("-p", "--path", help="input test case path",
                        type=str, default=None)

    # parser.add_argument("-o", "--outpath", help="output path",
    #                     type=str, default=None)

    args = parser.parse_args()

    return args

def run_bd(data):

    st = time.time()
    m = Benders_Decomposition(data)
    optval = m.optimize()
    ed = time.time()
    m.print_results()
    rst = m.get_results()

    return optval, (ed - st), rst

def check_answer(ans, optval, tol=1e-4):

    if np.abs(ans - optval) < tol:
        logging.info('The answer given is CORRECT!')
        return True
    else:
        logging.info('The correct answer is %.4f, but our answer is %.4f.' %\
                     (ans, optval))
        logging.info('The answer given is WRONG!')
        return False

def main(args):

    probdata = sio.loadmat(args.path)
    optval, ttime, rst = run_bd(probdata)

    logging.info('Total time is {}'.format(ttime))
    if probdata.get('ans', None) is not None:
        logging.info('Result is {}'.format(
            check_answer(probdata['ans'], optval)))

if __name__ == '__main__':

    np.set_printoptions(precision=3, linewidth=200)

    args = arg_parser()

    main(args)