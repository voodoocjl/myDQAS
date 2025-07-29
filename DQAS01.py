import os
import pickle
import numpy as np
# from tensorcircuit.applications.graphdata import regular_graph_generator
import networkx as nx
from typing import Any, Iterator
from collections import namedtuple
from matplotlib import pyplot as plt
import random

# 首先导入PyTorch相关的模块
import torch
from schemes import dqas_Scheme
from FusionModel import dqas_translator
from Arguments import Arguments

# 最后导入TensorFlow
import tensorflow as tf
import inspect

seed = 42
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

args = Arguments()
Graph = Any


def regular_graph_generator(d: int, n: int, weights: bool = False, seed=0) -> Iterator[Graph]:
    while True:
        g = nx.random_regular_graph(d, n,seed=seed)
        # no need to check edge number since it is fixed
        for e in g.edges:
            g[e[0]][e[1]]["weight"] = np.random.uniform() if weights else 1.0
        yield g

def preset_byprob(prob):
    preset = []
    p = prob.shape[0]
    c = prob.shape[1]
    for i in range(p):
        j = np.random.choice(np.arange(c), p=np.array(prob[i]))
        preset.append(j)
    return preset


def get_preset(stp):
    return tf.argmax(stp, axis=1)


def repr_op(element):
    if isinstance(element, str):
        return element
    if isinstance(element, list) or isinstance(element, tuple):
        return str(tuple([repr_op(e) for e in element]))
    if callable(element.__repr__):
        return element.__repr__()  # type: ignore
    else:
        return element.__repr__  # type: ignore


def get_var(name):
    """
    call in customized functions and grab variable within DQAF framework function by var name str

    :param name:
    :return:
    """
    return inspect.stack()[2][0].f_locals[name]


def record():
    return result(
        get_var("epoch"), get_var("cand_preset_repr"), get_var("avcost1").numpy(), get_var("test_acc")
    )


def qaoa_block_vag(gdata, ops, nnp, preset, repeat, enable):
    nnp = nnp.numpy()
    pnnp = []
    chosen_ops = []
    repeated_preset = preset * repeat
    for i, j in enumerate(repeated_preset):
        if 'u' in ops[j]:
            pnnp.append(nnp[i, j])
            chosen_ops.append(ops[j])
        else:
            pnnp.append(nnp[i, j][:, 0])
            chosen_ops.append(ops[j])
        # pnnp.append([nnp[i, j]])
        # chosen_ops.append(ops[j])
    edges = []
    for e in gdata.edges:
        edges.append(e)
    design = dqas_translator(chosen_ops, edges, repeat, 'full', enable)
    # pnnp = array_to_tensor(np.array(pnnp))  # complex
    # pnnp = tf.ragged.constant(pnnp, dtype=getattr(tf, cons.dtypestr))
    # design['pnnp'] = tf.ragged.constant(pnnp, dtype=dtype)
    design['pnnp'] = pnnp
    design['preset'] = preset
    design['edges'] = edges

    val_loss, model_grads, test_acc = dqas_Scheme(design, 'MNIST', 'init', 2)

    val_loss = tf.constant(val_loss, dtype=dtype)
    gr = tf.constant(model_grads, dtype=dtype)
    
    new_shape = [gr.shape[0] // 4, 4] + gr.shape[1:].as_list()
    gr = tf.reshape(gr, new_shape)
    gr = tf.transpose(gr, perm=[0, 2, 1, 3])
    # gr = design['pnnp'].with_values(gr)

    gmatrix = np.zeros_like(nnp)
    for j in range(gr.shape[0]):
        gmatrix[j, repeated_preset[j]] = gr[j][0]

    gmatrix = tf.constant(gmatrix)

    return val_loss, gmatrix, test_acc

def evaluate(gdata, ops,preset, nnp, repeat):
    nnp = nnp.numpy()
    chosen_ops = []
    pnnp = []
    repeated_preset = preset * repeat
    for i, j in enumerate(repeated_preset):
        if 'u' in ops[j]:
            pnnp.append(nnp[i, j])
            chosen_ops.append(ops[j])
        else:
            pnnp.append(nnp[i, j][:, 0])
            chosen_ops.append(ops[j])
        
    edges = []
    for e in gdata.edges:
        edges.append(e)
    design = dqas_translator(chosen_ops, edges, repeat, 'full', enable)
   
    design['pnnp'] = pnnp
    design['preset'] = preset
    design['edges'] = edges
    _, _, test_acc = dqas_Scheme(design, 'MNIST', 'init', 20)
    
    return test_acc


def DQAS_search(stp, nnp, epoch, enable):
    prob = tf.math.exp(stp) / tf.tile(
        tf.math.reduce_sum(tf.math.exp(stp), axis=1)[:, tf.newaxis], [1, category]
    )  # softmax categorical probability

    deri_stp = []
    deri_nnp = []
    avcost2 = 0
    costl = []
    test_acc_list = []

    if stp_regularization is not None:
        stp_penalty_gradient = stp_regularization(stp, nnp)
        if verbose:
            print("stp_penalty_gradient:", stp_penalty_gradient.numpy())
    else:
        stp_penalty_gradient = 0.0
    if nnp_regularization is not None:
        nnp_penalty_gradient = nnp_regularization(stp, nnp)
        if verbose:
            print("nnpp_penalty_gradient:", nnp_penalty_gradient.numpy())
    else:
        nnp_penalty_gradient = 0.0

    edges = None
    min_loss = 5
    for _, gdata in zip(range(batch), g):
        preset = preset_byprob(prob)
        if noise is not None:
            loss, gnnp, test_acc = qaoa_block_vag(gdata, op_pool, nnp + noise, preset, repeat, enable)
        else:
            loss, gnnp, test_acc = qaoa_block_vag(gdata, op_pool, nnp, preset, repeat, enable)

        gs = tf.tensor_scatter_nd_add(
            tf.cast(-prob, dtype=dtype),
            tf.constant(list(zip(range(p), preset))),
            tf.ones([p], dtype=dtype),
        )  # \nabla lnp
        deri_stp.append(
            (tf.cast(loss, dtype=dtype) - tf.cast(avcost2, dtype=dtype))
            * tf.cast(gs, dtype=dtype)
        )
        deri_nnp.append(gnnp)
        costl.append(loss.numpy())
        test_acc_list.append(test_acc)
        if loss.numpy() < min_loss:
            min_loss = loss.numpy()
            edges = [e for e in gdata.edges]

        avcost1 = tf.convert_to_tensor(np.min(costl))
        avtestacc = tf.convert_to_tensor(np.max(test_acc_list))

    print(
        "batched average loss: ",
        np.mean(costl),
        " batched loss std: ",
        np.std(costl),
        "\nmin_loss: ",
        avcost1.numpy(),  # type: ignore
    )

    batched_gs = tf.math.reduce_mean(
        tf.convert_to_tensor(deri_stp, dtype=dtype), axis=0
    )
    batched_gnnp = tf.math.reduce_mean(
        tf.convert_to_tensor(deri_nnp, dtype=dtype), axis=0
    )
    if verbose:
        print("batched gradient of stp: \n", batched_gs.numpy())
        print("batched gradient of nnp: \n", batched_gnnp.numpy())

    network_opt.apply_gradients(
        zip([batched_gnnp + nnp_penalty_gradient], [nnp])
    )
    structure_opt.apply_gradients(
        zip([batched_gs + stp_penalty_gradient], [stp])
    )
    if verbose:
        print(
            "strcuture parameter: \n",
            stp.numpy(),
            "\n network parameter: \n",
            nnp.numpy(),
        )

    cand_preset = get_preset(stp).numpy()
    cand_preset_repr = [repr_op(op_pool[f]) for f in cand_preset]
    print("best candidates so far:", cand_preset_repr)
    test_acc = evaluate(gdata, op_pool, cand_preset, nnp, repeat)

    return stp, nnp, record(), edges


if __name__ == '__main__':
    print("Starting DQAS initialization...")
    
    args = Arguments()
    p = 12

    repeat = 1

    # op_pool = ['rx', 'ry', 'rz', 'xx', 'yy', 'zz', 'u3', 'cu3']
    op_pool = ['I', 'data', 'u3', 'cu3']
    

    category = len(op_pool)
    g = regular_graph_generator(n=4, d=2, seed=seed)
    result = namedtuple("result", ["epoch", "cand", "loss", "test_acc"])

    verbose = None
    dtype = tf.float32
    batch = 10
    noise = None
    # noise = np.random.normal(loc=0.0, scale=0.2, size=[2 * repeat * p, c])
    # noise = tf.constant(noise, dtype=tf.float32)
    network_opt = tf.keras.optimizers.Adam(learning_rate=0.1)  # network
    structure_opt = tf.keras.optimizers.Adam(
        learning_rate=0.1, beta_1=0.8, beta_2=0.99
    )  # structure
    stp_regularization = None
    nnp_regularization = None

    epoch_init = 0
    n_qubits = 4
    nnp_initial_value = np.random.normal(loc=0.23, scale=0.06, size=[repeat * p, category, n_qubits, 3])      # [12, 4, 4, 3]
    stp_initial_value = np.zeros([p, category])
    history = []
    edges = []
    if os.path.isfile('step.history'):
        with open('step.history', 'rb') as f:
            stp_initial_value, nnp_initial_value, history, edges = pickle.load(f)
        epoch_init = len(history)

    nnp = tf.Variable(initial_value=nnp_initial_value, dtype=dtype)
    stp = tf.Variable(initial_value=stp_initial_value, dtype=dtype)

    enable = np.ones((repeat, p, args.n_qubits), dtype=np.bool_)
    # enable[0,1,1]=False
    avcost1 = 0
    dqas_epoch = 50

    print("DQAS setup completed, starting training...")
    
    try:
        for epoch in range(epoch_init, dqas_epoch):
            try:
                print("Epoch: ", epoch)
                stp, nnp, cur_history, edge = DQAS_search(stp, nnp, epoch, enable)
                history.append(cur_history)
                edges.append(edge)
                if len(history) > 20 and len(history) % 20 == 0:
                    cur_loss = [h.loss for h in history[-20:]]
                    last_loss = [h.loss for h in history[-40:-20]]
                    eta = abs(sum(cur_loss) / len(cur_loss) - sum(last_loss) / len(last_loss))
                    # if eta < 0.001:
                    #     raise Exception("stop iteration.")
            finally:
                with open('step.history', 'wb') as f:
                    pickle.dump((stp, nnp, history, edges), f)
    finally:
        epochs = np.arange(len(history))
        data = np.array([r.loss for r in history])
        plt.figure()
        plt.plot(epochs, data)
        plt.xlabel("epoch")
        plt.ylabel("objective (loss)")
        plt.savefig("loss_plot.pdf")
        plt.close()

        test_acc_data = np.array([r.test_acc for r in history])
        plt.figure()
        plt.plot(epochs, test_acc_data)
        plt.xlabel("epoch")
        plt.ylabel("test acc")
        plt.savefig("test_acc_plot.pdf")
        plt.close()

        with open('history.csv', 'w') as f:
            print('epoch, loss, test_acc, cand', file=f)
            for h in history:
                print(h.epoch, h.loss, h.test_acc, ' '.join(h.cand), sep=',', file=f)
