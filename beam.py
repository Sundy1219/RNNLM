import tensorflow as tf
import numpy as np


class BeamSearch():
    def __init__(self, predict, initial_state, prime_labels):
        """Initializes the beam search.

        Args:
            predict:
                A function that takes a `sample` and a `state`. It then performs
                the computation on the last word in `sample`.
            initial_state:
                The initial state of the RNN.
            prime_labels:
                A list of labels corresponding to the priming text. This must
                not be empty.
        """

        if not prime_labels:
            raise ValueError('prime_labels must be a non-empty list.')
        self.predict = predict
        self.initial_state = initial_state
        self.prime_labels = prime_labels

    def predict_samples(self, samples, states):
        probs = []
        next_states = []
        for i in range(len(samples)):
            prob, next_state = self.predict(samples[i], states[i])
            probs.append(prob.squeeze())
            next_states.append(next_state)
        return np.array(probs), next_states

    def search(self, oov, eos, k=1, maxsample=4000, use_unk=False):
        """Return k samples (beams) and their NLL scores.

        Each sample is a sequence of labels, either ending with `eos` or
        truncated to length of `maxsample`. `use_unk` allow usage of `oov`
        (out-of-vocabulary) label in samples
        """

        # A list of probabilities of our samples.
        probs = []

        prime_sample = []
        prime_score = 0
        prime_state = self.initial_state

        # Initialize the live sample with the prime.
        for i, label in enumerate(self.prime_labels):
            prime_sample.append(label)

            # The first word does not contribute to the score as the probs have
            # not yet been determined.
            if i > 0:
                prime_score = prime_score - np.log(probs[0, label])
            probs, prime_state = self.predict(prime_sample, prime_state)
        print np.shape(probs)
        print np.log(probs)
        dead_k = 0  # samples that reached eos
        dead_samples = []
        dead_scores = []
        dead_states = []

        live_k = 1  # samples that did not yet reached eos
        live_samples = [prime_sample]
        live_scores = [prime_score]
        live_states = [prime_state]
        print live_samples
        print live_scores
        print np.shape(live_states)
        while live_k and dead_k < k:
            # total score for every sample is sum of -log of word prb
            cand_scores = np.array(live_scores)[:, None] - np.log(probs)
            print cand_scores
            if not use_unk and oov is not None:
                cand_scores[:, oov] = 1e20
            cand_flat = cand_scores.flatten()
            print 'cand_flat is '
            print cand_flat
            print '\n'
            # find the best (lowest) scores we have from all possible samples and new words
            ranks_flat = cand_flat.argsort()[:(k - dead_k)]
            print 'cand_flat.argsort()'
            print cand_flat.argsort()
            print '\n'
            print 'ranks_flat is'
            print ranks_flat
            print '\n'
            live_scores = cand_flat[ranks_flat]
            print 'live_scores'
            print live_scores

            # append the new words to their appropriate live sample
            voc_size = probs.shape[1]
            print live_samples[412//voc_size]
            
            live_samples = [live_samples[r // voc_size] + [r % voc_size] for r in ranks_flat]
            print "live_samples is"
            print live_samples
            live_states = [live_states[r // voc_size] for r in ranks_flat]
            print 'live_states'
            print np.shape(live_states)
            # live samples that should be dead are...
            zombie = [s[-1] == eos or len(s) >= maxsample for s in live_samples]
            print 'zombie'
            print zombie
            # add zombies to the dead
            dead_samples += [s for s, z in zip(live_samples, zombie) if z]  # remove first label == empty
            dead_scores += [s for s, z in zip(live_scores, zombie) if z]
            dead_states += [s for s, z in zip(live_states, zombie) if z]
            dead_k = len(dead_samples)
            # remove zombies from the living
            live_samples = [s for s, z in zip(live_samples, zombie) if not z]
            live_scores = [s for s, z in zip(live_scores, zombie) if not z]
            live_states = [s for s, z in zip(live_states, zombie) if not z]
            live_k = len(live_samples)

            # Finally, compute the next-step probabilities and states.
            probs, live_states = self.predict_samples(live_samples, live_states)

        return dead_samples + live_samples, dead_scores + live_scores
