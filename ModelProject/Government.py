from math import tau
from types import SimpleNamespace

import numpy as np

from scipy import optimize

from Consumer import ConsumerClass

class GovernmentClass(ConsumerClass):
    """ a government raising revenue from the consumer in Consumer.py

    Two kinds of instrument:

    1) a lump-sum tax T, which reduces income
    2) product taxes tau1, tau2, tau3, which raise the prices

    The consumer is the ConsumerClass, so everything from there is inherited.

    """

    def __init__(self,par=None):

        # a. default setup
        self.setup()
        self.setup_government()

        # b. update parameters
        if not par is None:
            for k,v in par.items():
                self.par.__dict__[k] = v

        # c. remember the situation without taxes
        #    this must happen *after* b., or a changed price would not be picked up
        self.sync_pre_tax()

    def setup_government(self):
        """ add the tax instruments to the parameters """

        par = self.par

        # a. lump-sum tax
        par.T = 0.0 # lump-sum tax (a transfer if negative)

        # b. product taxes
        par.tau1 = 0.0 # tax rate on food
        par.tau2 = 0.0 # tax rate on bus trips
        par.tau3 = 0.0 # tax rate on train trips

    def sync_pre_tax(self):
        """ store the current prices and income as the situation without taxes

        Revenue is collected at these prices, so they must be the ones *before*
        any taxes were added.

        """

        par = self.par

        par.p1_pre = par.p1
        par.p2_pre = par.p2
        par.p3_pre = par.p3
        par.I_pre = par.I

    ##############################
    # 1. what the consumer faces #
    ##############################

    def set_taxes(self,T=0.0,tau1=0.0,tau2=0.0,tau3=0.0):
        """ set the taxes, and update the prices and the income the consumer faces

        The price the consumer pays for good j is (1+tau_j) times the price the
        seller receives, and income is reduced by the lump-sum tax. After this
        call every method inherited from ConsumerClass -- .solve(), .shares(),
        .value_of_choice(), .solve_grid() -- automatically refers to the
        situation *with* taxes.

        Args:

            T (float): lump-sum tax
            tau1 (float): tax rate on food
            tau2 (float): tax rate on bus trips
            tau3 (float): tax rate on train trips

        """

        par = self.par

        # a. remember the taxes
        par.T = T
        par.tau1 = tau1
        par.tau2 = tau2
        par.tau3 = tau3

        # b. the prices the consumer pays
        par.p1 = (1+tau1)*par.p1_pre
        par.p2 = (1+tau2)*par.p2_pre
        par.p3 = (1+tau3)*par.p3_pre

        # c. income after the lump-sum tax
        par.I = par.I_pre - T

    #########################################
    # 2. revenue, and what the consumer gets #
    #########################################

    def tax_revenue(self,opt=None):
        """ total tax revenue given the taxes currently set

        Note that revenue is collected at the prices the *seller* receives, so
        the tax paid on good j is tau_j*p_j_pre*x_j.

        Args:

            opt (SimpleNamespace): a solution from .solve(). Solved for here if
                not given -- pass it in when you already have it, to avoid
                solving the same problem twice

        Returns:

            (float): tax revenue

        """

        par = self.par

        # a. what does the consumer buy, given the taxes?
        #    .quantities() takes the nested shares (s1,w) from the solution
        if opt is None: opt = self.solve(do_print=False)

        x1,x2,x3 = self.quantities(opt.s1,opt.w)

        # b. the lump-sum tax, plus the product tax on each good
        R = par.T \
            + par.tau1*par.p1_pre*x1 \
            + par.tau2*par.p2_pre*x2 \
            + par.tau3*par.p3_pre*x3

        return R

    def revenue_and_utility(self,tau,goods=(2,)):
        """ revenue and utility when the same tax rate is put on each good in goods

        Args:

            tau (float): the common tax rate
            goods (tuple): which goods to tax, e.g. (2,) or (2,3) or (1,2,3)

        Returns:

            (tuple): (revenue, utility)

        """

        tau1 = tau if 1 in goods else 0.0
        tau2 = tau if 2 in goods else 0.0
        tau3 = tau if 3 in goods else 0.0

        self.set_taxes(T=0.0,tau1=tau1,tau2=tau2,tau3=tau3)

        opt = self.solve(do_print=False)
        R = self.tax_revenue(opt)
        u = opt.u

        return R,u

    def revenue_and_utility_lump_sum(self,T):
        """ the same, for a lump-sum tax of T

        Args:

            T (float): the lump-sum tax

        Returns:

            (tuple): (revenue, utility)

        """
        self.set_taxes(T=T,tau1=0.0,tau2=0.0,tau3=0.0)
        opt = self.solve(do_print=False)
        R = self.tax_revenue(opt)
        u = opt.u

        return R,u

    ##########################################
    # 3. hitting a given revenue requirement #
    ##########################################

    def max_revenue(self,goods=(2,),tau_max=10.0,N=1001):
        """ the largest revenue this instrument can ever raise

        A grid over the tax rate is enough, exactly as in section 2.1: compute
        the revenue in every grid point and keep the best one.

        If the answer comes back at tau_max, the curve was still rising when the
        grid ran out -- there is no top in the range searched.

        Args:

            goods (tuple): which goods to tax
            tau_max (float): largest tax rate to consider
            N (int): number of grid points

        Returns:

            (tuple): (the revenue-maximizing rate, the largest revenue)

        """

        # a. grid over the tax rate
        tau_vec = np.linspace(0,tau_max,N)

        # b. revenue in every grid point
        R_vec = np.empty(N)
        for i,tau_i in enumerate(tau_vec):
            R_vec[i],_ = self.revenue_and_utility(tau_i,goods=goods)

        # c. the best point
        i_max = np.argmax(R_vec)
        tau = tau_vec[i_max]
        R = R_vec[i_max]

        return tau,R

    def find_tax_rate(self,R_target,goods=(2,),bracket=(1e-10,1.0)):
        """ the tax rate on goods that raises exactly R_target

        Careful: revenue is not always increasing in the tax rate. There can be
        two rates that raise the same revenue, and a revenue target above the
        largest possible revenue cannot be reached at all. In that case there is
        no sign change in the bracket, and the root-finder will raise a
        ValueError -- which is the correct answer, not a bug. Catch it and
        return np.nan.

        Args:

            R_target (float): the revenue requirement
            goods (tuple): which goods to tax
            bracket (tuple): interval of tax rates to search in

        Returns:

            (float): the tax rate, or np.nan if the target cannot be reached

        """

        def f(tau_i):
            R,_ = self.revenue_and_utility(tau_i,goods=goods)
            return R - R_target

        try:
            res = optimize.root_scalar(f,bracket=bracket,method='brentq')
            tau = res.root
        except ValueError:
            tau = np.nan

        return tau
