import theano
import numpy
from theano import tensor as T

class GibbsSampler(object):
    def __init__(self, random, rbm):
        self.random = random
        self.rbm = rbm

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.rbm.W) + self.rbm.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.rbm.W.T) + self.rbm.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that random.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.random.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that random.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.random.normal(
                size=v1_mean.shape,
                avg=v1_mean,
                std=1.0,
                dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def update_visible_hidden_visible(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]

    def update_hidden_visible_hidden(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                    pre_sigmoid_h1, h1_mean, h1_sample]


class HamiltonianMonteCarloSampler(object):
    def __init__(self,
            random,
            rbm,
            initial_positions,
            initial_stepsize=1e-3,
            target_acceptance_rate=.65,
            n_steps=20,
            stepsize_decrement=0.98,
            minimum_stepsize=0.001,
            maximum_stepsize=0.25,
            stepsize_increment=1.02,
            avg_acceptance_slowness=0.9):
        self.random = random
        self.rbm = rbm

        self.energy_fn = self.rbm.free_energy

        # allocate shared variables
        stepsize = theano.shared(numpy.cast[theano.config.floatX](initial_stepsize), 'hmc_stepsize')
        avg_acceptance_rate = theano.shared(target_acceptance_rate,
                                      'avg_acceptance_rate')

        self.positions = theano.shared(initial_positions, 'positions')
        self.stepsize = stepsize
        self.minimum_stepsize = minimum_stepsize
        self.maximum_stepsize = maximum_stepsize
        self.avg_acceptance_rate = avg_acceptance_rate
        self.target_acceptance_rate = target_acceptance_rate

        # define graph for an `n_steps` HMC simulation
        accept, final_position = self.move(
            self.positions,
            stepsize,
            n_steps)

        # define the dictionary of updates, to apply on every `simulate` call
        simulate_updates = self.updates(
            self.positions,
            stepsize,
            avg_acceptance_rate,
            final_position=final_position,
            accept=accept,
            minimum_stepsize=minimum_stepsize,
            maximum_stepsize=maximum_stepsize,
            stepsize_increment=stepsize_increment,
            stepsize_decrement=stepsize_decrement,
            target_acceptance_rate=target_acceptance_rate,
            avg_acceptance_slowness=avg_acceptance_slowness)

        self._updates = simulate_updates

        # compile theano function
        self.simulate = theano.function([], [], updates=simulate_updates)

    def draw(self):
        self.simulate()
        return self.positions.get_value(borrow=False)

    def leapfrog(self, position, velocity, step):
        """
        Performs one step of leapfrog update using Hamiltonian dynamics.

        Parameters
        ----------
        position: theano matrix
            position at time t
        velocity: theano matrix
            represents velocity at time t - stepsize/2
        step: theano scalar
            scalar value for controlling amount by which to move

        Returns
        ----------
        value1: [theano matrix, theano matrix]
            symbolic theano matrices for position at t+stepsize ad
            velocit at t + stepsize/2
        value2: dictionary
            Dictionary of updates for the scan operation
        """
        dE_dpos = T.grad(self.energy_fn(position).sum(), position)
        new_velocity = velocity - step * dE_dpos
        new_position = position + step * new_velocity
        return [new_position, new_velocity], {}

    def simulate_dynamics(self, initial_position, initial_velocity, stepsize, n_steps):
        initial_energy = self.energy_fn(initial_position)
        dE_dpos = T.grad(initial_energy.sum(), initial_position)
        velocity_half_step = initial_velocity - 0.5 * stepsize * dE_dpos
        position_full_step = initial_position + stepsize * velocity_half_step

        (all_positions, all_velocities), scan_updates = theano.scan(
                self.leapfrog,
                outputs_info=[
                    dict(initial=position_full_step),
                    dict(initial=velocity_half_step)
                ],
                non_sequences=[stepsize],
                n_steps=n_steps - 1)
        final_position = all_positions[-1]
        final_velocity = all_velocities[-1]

        assert(not scan_updates)

        energy = self.energy_fn(final_position)
        final_velocity = final_velocity - 0.5 * stepsize * T.grad(energy.sum(), final_position)
        return final_position, final_velocity

    def move(self, positions, stepsize, n_steps):
        initial_velocity = self.random.normal(size=positions.shape)
        final_position, final_velocity = self.simulate_dynamics(
                initial_position=positions,
                initial_velocity=initial_velocity,
                stepsize=stepsize,
                n_steps=n_steps)
        accept = self.metropolis_hastings_accept(
                previous_energy=self.hamiltonian(positions, initial_velocity),
                energy_next=self.hamiltonian(final_position, final_velocity)
            )
        return accept, final_position

    def metropolis_hastings_accept(self, previous_energy, energy_next):
        energy_delta = previous_energy - energy_next
        return (T.exp(energy_delta) - self.random.uniform(size=previous_energy.shape)) >= 0

    def hamiltonian(self, position, velocity):
        return self.energy_fn(position) + self.kinetic_energy(velocity)

    def kinetic_energy(self, velocity):
        return 0.5 * (velocity ** 2).sum(axis=1)

    def updates(self,
            positions,
            stepsize,
            avg_acceptance_rate,
            final_position,
            accept,
            target_acceptance_rate,
            stepsize_increment,
            stepsize_decrement,
            minimum_stepsize,
            maximum_stepsize,
            avg_acceptance_slowness):

        accept_matrix = accept.dimshuffle(0, *(('x',) * (final_position.ndim - 1)))
        new_positions = T.switch(accept_matrix, final_position, positions)

        mean_dtype = theano.scalar.upcast(accept.dtype, avg_acceptance_rate.dtype)
        new_acceptance_rate = T.add(
                avg_acceptance_slowness * avg_acceptance_rate,
                (1.0 - avg_acceptance_slowness) * accept.mean(dtype=mean_dtype))

        # STEPSIZE UPDATES #
        # if acceptance rate is too low, our sampler is too "noisy" and we reduce
        # the stepsize. If it is too high, our sampler is too conservative, we can
        # get away with a larger stepsize (resulting in better mixing).
        _new_stepsize = T.switch(avg_acceptance_rate > target_acceptance_rate,
                                  stepsize * stepsize_increment, stepsize * stepsize_decrement)

        # maintain stepsize in [minimum_stepsize, maximum_stepsize]
        new_stepsize = T.clip(_new_stepsize, minimum_stepsize, maximum_stepsize)

        return [(positions, new_positions),
            (stepsize, new_stepsize),
            (avg_acceptance_rate, new_acceptance_rate)]










