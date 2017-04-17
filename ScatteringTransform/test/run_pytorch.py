import torch
import filters_bank_pytorch as filters_bank
import scattering_pytorch as scattering


def run_filter_bank(M, N, J):

    filters = filters_bank.filters_bank(M, N, J)
    d_save = {}
    # Save phi
    d_save["phi"] = {}
    for key in filters["phi"].keys():
        val = filters["phi"][key]
        if isinstance(val, torch.FloatTensor):
            val_numpy = val.cpu().numpy()
            d_save["phi"][key] = val_numpy
    # Save psi
    d_save["psi"] = []
    for elem in filters["psi"]:
        d = {}
        for key in elem.keys():
            val = elem[key]
            if isinstance(val, torch.FloatTensor):
                val_numpy = val.cpu().numpy()
                d[key] = val_numpy
        d_save["psi"].append(d)

    return d_save


def run_scattering(X, use_cuda=False):

    # Ensure NCHW format
    assert X.shape[1] < min(X.shape[2:])

    M, N = X.shape[2:]

    if use_cuda:

        scat = scattering.Scattering(M=M, N=N, J=2, check=True).cuda()
        list_S = scat.forward(torch.FloatTensor(X).cuda())

    else:
        scat = scattering.Scattering(M=M, N=N, J=2, check=True)
        list_S = scat.forward(torch.FloatTensor(X))

    return list_S
