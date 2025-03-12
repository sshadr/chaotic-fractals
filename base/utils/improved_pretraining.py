import torch
import numpy as np
from easydict import EasyDict as edict

def fractal_generator(num_matrices, seed):
    torch.random.manual_seed(seed)
    system = sample_system(num_matrices)#, beta=(4.5, 5))
    cpu_code = fractaldb_to_ifs_code(system)

    # cpu_code is a dictionary 
    return cpu_code

def fractaldb_to_ifs_code(system):
    num_matrices = system.shape[0]
    m = []
    t = []
    dets = []
    for channel in range(num_matrices):
        mat = torch.tensor(system[channel, :, :2].reshape(2, 2), dtype=torch.float32)
        tr = torch.tensor(system[channel, :, 2].reshape(2), dtype=torch.float32)
        determinant = torch.linalg.det(mat)
        m.append(mat.tolist())
        t.append(tr.tolist())
        dets.append(determinant.item())

    p_det = np.abs(np.array(dets)) + 0.1
    p_det = p_det/p_det.sum()
    
    d = edict()
    d.ifs_m = m
    d.ifs_t = t
    d.ifs_p = p_det.tolist()
    
    return d

# The following are from 
# https://github.com/catalys1/fractal-pretraining/blob/main/fractal_learning/fractals/ifs.py

def sample_svs(n, a, rng=None):
    '''Sample singular values. 2*`n` singular values are sampled such that the following conditions
    are satisfied, for singular values sv_{i} and i = 0, ..., 2n-1:
    
    1. 0 <= sv_{i} <= 1
    2. sv_{2i} >= sv_{2i+1}
    3. w.T @ S = `a`, for S = [sv_{0}, ..., sv_{2n-1}] and w = [1, 2, ..., 1, 2]
    
    Args:
        n (int): number of pairs of singular values to sample.
        a (float): constraint on the weighted sum of all singular values. Note that a must be in the
            range (0, 3*n).
        rng (Optional[numpy.random._generator.Generator]): random number generator. If None (default), it defaults
            to np.random.default_rng().
            
    Returns:
        Numpy array of shape (n, 2) containing the singular values.
    '''
    if rng is None: rng = np.random.default_rng()
    if a < 0: a == 0
    elif a > 3*n: a == 3*n
    s = np.empty((n, 2))
    p = a
    q = a - 3*n + 3
    # sample the first 2*(n-1) singular values (first n-1 pairs)
    for i in range(n - 1):
        s1 = rng.uniform(max(0, q/3), min(1, p))
        q -= s1
        p -= s1
        s2 = rng.uniform(max(0, q/2), min(s1, p/2))
        q = q - 2 * s2 + 3
        p -= 2 * s2
        s[i, :] = s1, s2
    # sample the last pair of singular values
    s2 = rng.uniform(max(0, (p-1)/2), p/3)
    s1 = p - 2*s2
    s[-1, :] = s1, s2
    
    return s

def sample_svs_rej(n, a, rng=None):
    '''Sample singular values uniformly from the joint distribution over the n-dimensional surface
    defined by the constraints (see sample_svs). Uniform sampling is achieved by means of rejection
    sampling.

    Args:
        n (int): number of pairs of singular values to sample.
        a (float): constraint on the weighted sum of all singular values. Note that a must be in the
            range (0, 3*n).
        rng (Optional[numpy.random._generator.Generator]): random number generator. If None (default), it defaults
            to np.random.default_rng().
            
    Returns:
        Numpy array of shape (n, 2) containing the singular values.
    '''
    if rng is None:
        rng = np.random.default_rng()
    if a < 0: a = 0
    elif a > 3 * n: a = 3 * n

    w = np.ones(2 * n - 1)
    w[1::2] = 2
    s = np.zeros((n, 2))
    for i in range(1000):
        s.ravel()[:-1] = rng.random(2 * n - 1)
        # restrict to below the y=x line
        r = s[:, 1] > s[:, 0]
        s[r, :] = s[r][:, ::-1]
        # check if valid or reject
        b = (a - w @ s.ravel()[:-1]) / 2
        if b <= s[-1, 0] and b >= 0:
            s[-1, 1] = b
            break
    else:
        print('Rejection sampling failed')
    return s

def sample_system(n=None, constrain=True, bval=1, rng=None, beta=None, sample_fn=None):
    '''Return n random affine transforms. If constrain=True, enforce the transforms
    to be strictly contractive (by forcing singular values to be less than 1).
    
    Args:
        n (Union[range,Tuple[int,int],List[int,int],None]): range of values to sample from for the number of
            transforms to sample. If None (default), then sample from range(2, 8).
        constrain (bool): if True, enforce contractivity of transformations. Technically, an IFS must be
            contractive; however, FractalDB does not enforce it during search, so it is left as an option here.
            Default: True.
        bval (Union[int,float]): maximum magnitude of the translation parameters sampled for each transform.
            The translation parameters don't effect contractivity, and so can be chosen arbitrarily. Ignored and set
            to 1 when constrain is False. Default: 1.
        rng (Optional[numpy.random._generator.Generator]): random number generator. If None (default), it defaults
            to np.random.default_rng().
        beta (float or Tuple[float, float]): range for weighted sum of singular values when constrain==True. Let 
            q ~ U(beta[0], beta[1]), then we enforce $\sum_{i=0}^{n-1} (s^i_1 + 2*s^i_2) = q$.
        sample_fn (callable): function used for sampling singular values. Should accept three arguments: n, for
            the size of the system; a, for the sigma-factor; and rng, the random generator. When None (default),
            uses sample_svs.
    
    Returns:
        Numpy array of shape (n, 2, 3), containing n sets of 2x3 affine transformation matrices.
        '''
    if rng is None:
        rng = np.random.default_rng()
    if n is None:
        n = rng.integers(2, 8)
    elif isinstance(n, range):
        n = rng.integers(n.start, n.stop)
    elif isinstance(n, (tuple, list)):
        n = rng.integers(*n)
        
    if beta is None:
        beta = ((5 + n) / 2, (6 + n) / 2)

    if sample_fn is None:
        sample_fn = sample_svs
        
    if constrain:
        # sample a matrix with singular values < 1 (a contraction)
        # 1. sample the singular vectors--random orthonormal matrices--by randomly rotating the standard basis
        base = np.sign(rng.random((2*n, 2, 1)) - 0.5) * np.eye(2)
        angle = rng.uniform(-np.pi, np.pi, 2*n)
        ss = np.sin(angle)
        cc = np.cos(angle)
        rmat = np.empty((2 * n, 2, 2))
        rmat[:, 0, 0] = cc
        rmat[:, 0, 1] = -ss
        rmat[:, 1, 0] = ss
        rmat[:, 1, 1] = cc
        uv = rmat @ base
        u, v = uv[:n], uv[n:]
        # 2. sample the singular values
        a = rng.uniform(*beta)
        s = sample_fn(n, a, rng)
        # 3. sample the translation parameters from Uniform(-bval, bval) and create the transformation matrix
        m = np.empty((n, 2, 3))
        m[:, :, :2] = u * s[:, None, :] @ v
        m[:, :, 2] = rng.uniform(-bval, bval, (n, 2))
    else:
        m = rng.uniform(-1, 1, (n, 2, 3))

    return m