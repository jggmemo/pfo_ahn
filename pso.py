# encoding=utf8
import random
import numpy
from utility import Utility

__all__ = ['ParticleSwarmAlgorithm']


class ParticleSwarmAlgorithm(object):
    r"""Implementation of Particle Swarm Optimization algorithm.

    **Algorithm:** Particle Swarm Optimization algorithm

    **Date:** 2018

    **Authors:** Lucija Brezočnik, Grega Vrbančič, and Iztok Fister Jr.

    **License:** MIT

    **Reference paper:**
        Kennedy, J. and Eberhart, R. "Particle Swarm Optimization".
        Proceedings of IEEE International Conference on Neural Networks.
        IV. pp. 1942--1948, 1995.
    """

    def __init__(self, D, vMin, vMax, benchmark, nFES=1000, w=-0.35, NP=60, C1=-0.72, C2=2.029):
        r"""**__init__(self, NP, D, nFES, C1, C2, w, vMin, vMax, benchmark)**.

        Arguments:
            NP {integer} -- population size

            D {integer} -- dimension of problem

            nFES {integer} -- number of function evaluations

            C1 {decimal} -- cognitive component

            C2 {decimal} -- social component

            w {decimal} -- inertia weight

            vMin {decimal} -- minimal velocity

            vMax {decimal} -- maximal velocity

            benchmark {object} -- benchmark implementation object

        """

        self.benchmark = Utility().get_benchmark(benchmark)
        self.NP = NP  # population size; number of search agents
        self.D = D  # dimension of the problem
        self.C1 = C1  # cognitive component
        self.C2 = C2  # social component
        self.w = w  # inertia weight
        self.vMin = vMin  # minimal velocity
        self.vMax = vMax  # maximal velocity
        self.Lower = self.benchmark.Lower  # lower bound
        self.Upper = self.benchmark.Upper  # upper bound
        self.nFES = nFES  # number of function evaluations
        self.eval_flag = True  # evaluations flag
        self.evaluations = 0  # evaluations counter
        self.Fun = self.benchmark.function()

        self.Solution = numpy.zeros((self.NP, self.D))  # positions of search agents
        self.Velocity = numpy.zeros((self.NP, self.D))  # velocities of search agents

        self.pBestFitness = numpy.zeros(self.NP)  # personal best fitness
        self.pBestFitness.fill(float("inf"))
        self.pBestSolution = numpy.zeros((self.NP, self.D))  # personal best solution

        self.gBestFitness = float("inf")  # global best fitness
        self.gBestSolution = numpy.zeros(self.D)  # global best solution
        self.gBestAhn = []

    def init(self):
        """Initialize positions."""
        for i in range(self.NP):
            for j in range(self.D):
                self.Solution[i][j] = random.random() * \
                    (self.Upper[j] - self.Lower[j]) + self.Lower[j]

    def eval_true(self):
        """Check evaluations."""

        if self.evaluations == self.nFES:
            self.eval_flag = False

    def bounds(self, position):
        """Keep it within bounds."""
        for i in range(self.D):
            if position[i] < self.Lower[i]:
                position[i] = self.Lower[i]
            if position[i] > self.Upper[i]:
                position[i] = self.Upper[i]
        return position

    def move_particles(self):
        """Move particles in search space."""
        self.init()

        while self.eval_flag is not False:
            for i in range(self.NP):
                self.Solution[i] = self.bounds(self.Solution[i])

                self.eval_true()
                if self.eval_flag is not True:
                    break

                Fit, ahn = self.Fun(self.D, self.Solution[i])
                self.evaluations = self.evaluations + 1

                if Fit < self.pBestFitness[i]:
                    self.pBestFitness[i] = Fit
                    self.pBestSolution[i] = self.Solution[i]

                if Fit < self.gBestFitness:
                    self.gBestFitness = Fit
                    self.gBestSolution = self.Solution[i]
                    self.gBestAhn = ahn

            for i in range(self.NP):
                for j in range(self.D):
                    self.Velocity[i][j] = (self.w * self.Velocity[i][j]) + \
                        (self.C1 * random.random() * (self.pBestSolution[i][j] - self.Solution[i][j])) + \
                        (self.C2 * random.random() * (self.gBestSolution[j] - self.Solution[i][j]))

                    if self.Velocity[i][j] < self.vMin:
                        self.Velocity[i][j] = self.vMin
                    if self.Velocity[i][j] > self.vMax:
                        self.Velocity[i][j] = self.vMax

                    self.Solution[i][j] = self.Solution[i][j] + \
                        self.Velocity[i][j]

        return self.gBestFitness

    def run(self):
        """Run."""
        return self.move_particles()
